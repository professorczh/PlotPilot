#!/bin/bash

# ============================================================
#  PlotPilot（墨枢）- AI 小说创作平台 macOS 启动器
# ============================================================
#  一键即用，全自动流程：
#    ① 自动检测 uv 与 Python 环境
#    ② 自动创建虚拟环境 / 闪电安装依赖
#    ③ 清理残留端口，启动后端 GUI 控制中心
# ============================================================

# ANSI 颜色配置
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0;62m' # No Color
BOLD='\033[1m'

# 定位项目根目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/.."

echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "${BLUE}${BOLD}      PlotPilot（墨枢）- AI 小说创作平台 macOS 启动器${NC}"
echo -e "${BLUE}${BOLD}============================================================${NC}"
echo ""

# ════════════════════════════════════
# Step 1: 检测 Python 环境 与 uv 包管理器
# ════════════════════════════════════
echo -e "${BLUE}🔍 正在检查本地环境...${NC}"

# A) 优先检测 uv
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓ 发现包管理器 uv (极速安装器)${NC}"
    USE_UV=true
else
    echo -e "${YELLOW}! 未发现 uv，建议运行 'curl -LsSf https://astral.sh/uv/install.sh | sh' 安装以获得 10x 依赖安装速度${NC}"
    USE_UV=false
fi

# B) 检测 Python 3
if command -v python3 &> /dev/null; then
    PYTHON_SYS="python3"
    PYTHON_VER=$(python3 --version 2>&1)
    echo -e "${GREEN}✓ 发现系统 Python: $PYTHON_VER${NC}"
else
    echo -e "${RED}[ERROR] 未在 PATH 中找到 Python 3！请先安装 Python 3.10+。${NC}"
    echo -e "${YELLOW}下载地址: https://www.python.org/downloads/${NC}"
    exit 1
fi

# ════════════════════════════════════
# Step 2: 确保虚拟环境存在并就绪
# ════════════════════════════════════
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚙️ 首次运行：正在创建 Python 虚拟环境...${NC}"
    if [ "$USE_UV" = true ]; then
        uv venv .venv
    else
        $PYTHON_SYS -m venv .venv
    fi
    echo -e "${GREEN}✓ 虚拟环境创建成功 (.venv)${NC}"
fi

# 定位虚拟环境中的 Python 路径
VENV_PYTHON=".venv/bin/python"

# ════════════════════════════════════
# Step 3: 安装依赖包
# ════════════════════════════════════
echo -e "${BLUE}📦 正在检查/同步依赖项...${NC}"
if [ "$USE_UV" = true ]; then
    uv pip install -r requirements.txt
else
    $VENV_PYTHON -m pip install --upgrade pip
    $VENV_PYTHON -m pip install -r requirements.txt
fi
echo -e "${GREEN}✓ 依赖项更新完成${NC}"

# ════════════════════════════════════
# Step 4: 确保目录存在
# ════════════════════════════════════
mkdir -p logs
mkdir -p data/chromadb
mkdir -p data/logs

# ════════════════════════════════════
# Step 5: 清理占用端口的孤儿进程
# ════════════════════════════════════
for PORT in 8005 8006; do
    PID=$(lsof -t -i :$PORT -sTCP:LISTEN 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}🧹 发现端口 $PORT 被进程 $PID 占用，自动清理...${NC}"
        kill -9 $PID &>/dev/null
    fi
done

# ════════════════════════════════════
# Step 6: 启动统一 GUI 中心 (Hub)
# ════════════════════════════════════
echo ""
echo -e "${GREEN}${BOLD}🎉 环境准备就绪！正在启动 PlotPilot 统一控制中心...${NC}"
echo -e "${BLUE}控制中心窗口已分离运行，您可以最小化它，请勿直接关闭窗口。${NC}"
echo ""

# 使用虚拟环境的 Python 启动 Tkinter hub.py，并在后台脱离终端
# 这与 Windows 的 start "" pythonw.exe 核心魔法对齐，使得控制台可以安全关闭
nohup $VENV_PYTHON scripts/install/hub.py >logs/hub_stdout.log 2>logs/hub_stderr.log &

# 稍等片刻，给出启动提示
sleep 1.5
echo -e "${GREEN}✓ 启动信号已发出。您可以在浏览器访问：${BOLD}http://127.0.0.1:8005${NC}"
echo -e "${BLUE}实时运行日志可查看: logs/hub_stderr.log 和 logs/hub_stdout.log${NC}"
echo ""
exit 0
