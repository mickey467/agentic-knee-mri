"""Stdio runners proving each MCP server speaks real MCP over a transport.

Usage: python -m app.services.mcp <dicom|model|vis|rag>
The agent itself uses in-process @tool adapters (same logic, no transport
overhead); these entrypoints keep the "4 MCP servers" architecture genuine
and usable by any MCP client.
"""

import asyncio
import sys

SERVERS = ("dicom", "model", "vis", "rag")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in SERVERS:
        print(f"Usage: python -m app.services.mcp <{'|'.join(SERVERS)}>", file=sys.stderr)
        raise SystemExit(2)
    which = sys.argv[1]
    if which == "dicom":
        from app.services.mcp.dicom import create_dicom_mcp_server as factory
    elif which == "model":
        from app.services.mcp.model import create_model_mcp_server as factory
    elif which == "vis":
        from app.services.mcp.visualization import create_vis_mcp_server as factory
    else:
        from app.services.mcp.rag import create_rag_mcp_server as factory
    asyncio.run(factory().run_stdio_async())


if __name__ == "__main__":
    main()
