from pathlib import Path
import logging
from functools import lru_cache
from typing import Callable, Optional
import click
from fastmcp import FastMCP


# Configure verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class KnowledgeTool:
    def __init__(self, folder: Path):
        self.folder = folder

    @lru_cache(maxsize=128)
    def get_knowledge(self, article: str) -> str:
        """Read a knowledge resource file from the wiki dataset.

        Args:
            article: The file name of the article to read.

        Returns:
            The text contents of the requested knowledge article.

        Raises:
            ValueError: If the requested article file does not exist.
        """
        file_path = self.folder / article
        if not file_path.exists():
            raise ValueError(f"Article '{article}' not found")

        return file_path.read_text()


"""
MCP Environment Server - Exposes tau2 Environment tools via Model Context Protocol (MCP).

This module provides an MCP server implementation that wraps a tau2 Environment,
making its tools available through the MCP protocol using FastMCP.
"""


class MCPServer:
    """
    An MCP server that exposes the tools of an Environment via the Model Context Protocol.
    
    This server uses FastMCP to create an MCP-compatible interface for tau2 environments,
    allowing AI assistants to discover and use environment tools through the MCP protocol.
    """

    def __init__(self, server_name: str):
        logger.info("Initializing MCPEnvironmentServer")
        self.server_name = server_name
        logger.info(f"Server name: {self.server_name}")
        
        # Create FastMCP instance
        logger.debug("Creating FastMCP instance")
        self.mcp = FastMCP(
            name=self.server_name,
            instructions="bla",
        )
        logger.info("FastMCP instance created successfully")
        
        # Register tools
        logger.info("Starting tool registration")
        logger.info("MCPEnvironmentServer initialization complete")


    def register_tool(self, tool_fn: Callable):
        self.mcp.tool(tool_fn)

    def run(self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765):
        """
        Run the MCP server.

        Args:
            transport: The transport to use. Options:
                      - "stdio": Standard input/output (default for MCP)
                      - "http": HTTP with Server-Sent Events
            host: Host to bind to (only for HTTP transport)
            port: Port to bind to (only for HTTP transport)
        """
        if transport == "stdio":
            self.mcp.run()
        elif transport == "http":
            # Run with HTTP transport (SSE)
            self.mcp.run(transport="http", host=host, port=port)
        else:
            raise ValueError(f"Unknown transport: {transport}. Use 'stdio' or 'http'.")


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option("--port", default=8765, type=int, help="Port to bind to.")
@click.option("--folder", required=True, type=click.Path(exists=True, path_type=Path), help="Path to the knowledge wiki folder.")
def main(host: str, port: int, folder: Path):
    knowledge_tool = KnowledgeTool(folder)

    mcp_server = MCPServer(
        server_name="wiki_knowledge_server",
    )
    mcp_server.register_tool(knowledge_tool.get_knowledge)

    print(f"Starting MCP server on http://{host}:{port}")
    print("Server is ready to accept MCP protocol messages over HTTP.")
    print(f"MCP endpoint: http://{host}:{port}/mcp")
    mcp_server.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    main()
