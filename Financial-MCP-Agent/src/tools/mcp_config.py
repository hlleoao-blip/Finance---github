"""
MCP服务器配置模块 - 包含连接A股MCP服务器的配置信息
"""

import sys
from pathlib import Path

# 使用当前已验证的 Python 解释器启动 MCP server，避免 uv 依赖用户全局缓存目录
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
A_SHARE_MCP_DIR = WORKSPACE_ROOT / "a-share-mcp-is-just-i-need"

SERVER_CONFIGS = {
    "a_share_mcp_v2": {  
        "command": sys.executable,
        "args": [
            "-X",
            "utf8",
            "mcp_server.py"  # MCP服务器脚本
        ],
        "cwd": str(A_SHARE_MCP_DIR),  # A股 MCP 服务器项目路径
        "transport": "stdio",
    }
}
