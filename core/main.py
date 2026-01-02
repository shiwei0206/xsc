
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import time
import asyncio
from brain import DualBrain, GraphManager 


# ==================== 数据模型 ====================
class ChatMessage(BaseModel):
    sessionId: str
    content: str
    timestamp: Optional[int] = None


# ==================== 应用初始化 ====================
app = FastAPI(
    title="EvoMind API",
    version="2.0",
    description=""
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    brain = DualBrain()
    print("✅ 双脑系统 (DeepSeek + Neo4j) 初始化成功")
except Exception as e:
    print(f"❌ 系统初始化失败，请检查 .env 配置: {e}")

session_prompts: Dict[str, str] = {}



@app.get("/")
async def root():
    """服务状态检查"""
    return {
        "service": "EvoMind API (Real Mode)",
        "status": "running",
        "backend": "Neo4j + DeepSeek",
        "endpoints": {
            "chat": "POST /api/chat",
            "stream": "GET /api/stream?sessionId=xxx",
            "snapshot": "GET /api/graph/snapshot?sessionId=xxx"
        }
    }


@app.post("/api/chat")
async def process_chat(message: ChatMessage):
    """
    阶段一：接收用户 Prompt
    将 Prompt 存入内存，准备给 SSE 连接使用
    """
    if not message.content.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    print(f"[API] 收到消息 ({message.sessionId}): {message.content[:50]}...")

    # 存入 Session 存储
    session_prompts[message.sessionId] = message.content

    return {
        "code": 200,
        "message": "Prompt received",
        "data": {
            "sessionId": message.sessionId,
            "streamUrl": f"/api/stream?sessionId={message.sessionId}"
        }
    }


@app.get("/api/graph/snapshot")
async def get_graph_snapshot(sessionId: Optional[str] = None):
    """
    获取当前图谱快照
    用于页面初始化时加载已有的图谱数据
    """
    try:
        driver = brain.graph_manager.driver
        nodes = []
        links = []

        with driver.session() as session:
            # 查节点
            result_n = session.run("MATCH (n:Entity) RETURN n LIMIT 100")
            for record in result_n:
                node = record["n"]
                nodes.append({
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "group": node.get("group", "Concept"),
                    "color": node.get("color", "#cccccc"),
                    "val": 15
                })

            # 查关系
            result_r = session.run("MATCH (s:Entity)-[r]->(t:Entity) RETURN s.id, t.id, r.name LIMIT 100")
            for record in result_r:
                links.append({
                    "source": record["s.id"],
                    "target": record["t.id"],
                    "relationship": record["r.name"],
                    "width": 2
                })

        graph_data = {"nodes": nodes, "links": links}

        return JSONResponse(content={
            "code": 200,
            "data": graph_data,
            "message": "图谱快照获取成功",
            "stats": {
                "nodes": len(nodes),
                "links": len(links)
            }
        })

    except Exception as e:
        print(f"获取快照失败: {e}")
        return JSONResponse(content={
            "code": 500,
            "message": str(e),
            "data": {"nodes": [], "links": []}
        }, status_code=500)


@app.get("/api/stream")
async def stream_updates(sessionId: str):
    """
    阶段二：SSE 流式输出
    调用 DualBrain 核心逻辑，返回生成器
    """

    user_prompt = session_prompts.get(sessionId)

    if not user_prompt:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'control', 'status': 'error', 'payload': 'Session expired or invalid'})}\n\n"

        return StreamingResponse(error_gen(), media_type="text/event-stream")
    return StreamingResponse(
        brain.think(sessionId, user_prompt),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


# ==================== 清理资源 ====================
@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时断开数据库连接"""
    brain.close()


# ==================== 运行应用 ====================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )