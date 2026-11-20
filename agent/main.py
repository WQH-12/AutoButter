import sys

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

import my_action
import my_reco
import registry


def main():
    # Android 上工作目录是资源解包根；init_option 只影响日志落点，
    # 失败不该把整个 agent 拖死
    try:
        Toolkit.init_option("./")
    except Exception as exc:  # noqa: BLE001
        print("[agent] Toolkit.init_option failed: %s" % exc, flush=True)

    if len(sys.argv) < 2:
        print("Usage: python main.py <socket_id>")
        print("socket_id is provided by AgentIdentifier.")
        sys.exit(1)

    socket_id = sys.argv[-1]

    # 自定义识别必须在 start_up 之前注册：Maa 的 AgentClient 在 start_up 时
    # 就连上来取注册表，之后再注册不会同步到客户端。
    # 注册表集中在 registry.py，进程内模式和子进程模式共用同一份清单。
    print("[agent] register:", registry.register_agent_server(), flush=True)

    AgentServer.start_up(socket_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
