import json
import os
from dataclasses import dataclass, field
from typing import Optional

# from jupyter_lsp.specs import yaml
import yaml

from docuverse.utils import get_config_dir
from docuverse.utils import read_config_file
from docuverse.utils.config_resolver import resolve_optional

@dataclass
class Server:
    host: str = field(
        default="localhost",
        metadata={
            "help": "The hostname of the server to connect to"
        }
    )

    port: int = field(
        default=None,
        metadata={
            "help": "The port the server is listening on"
        }
    )

    user: Optional[str] = field(
        default=None,
        metadata={
            "help": "The username to connect with"
        }
    )

    password: Optional[str] = field(
        default=None,
        metadata={
            "help": "The password to connect with"
        }
    )

    api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "The API key to connect with"
        }
    )

    ssl_fingerprint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The SSL fingerprint of the server."
        }
    )

    server_pem_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The SSL certificate of the server."
        }
    )

    secure: bool = field(
        default=False,
        metadata={
            "help": "Whether the server is secure."
        }
    )

    server_name: Optional[str] = field(
        default="localhost",
        metadata={
            "help": "The name of the server (used for Milvus)."
        }
    )

    type: Optional[str] = field(
        default="http",
        metadata={
            "help": "The type of the server (can be either http or file)."
        }
    )

    def __post_init__(self):
        if self.host.find("file:") >= 0:
            self.type="file"
            self.server_name = self.host.replace("file:", "")

    def get(self, key:str, default:str|None=None):
        return getattr(self, key, default)


class RetrievalServers:
    def __init__(self, config: str):
        # ``config`` is the registry path (e.g. ``"servers/milvus_servers.json"``
        # or the legacy ``"config/milvus_servers.json"``). Route through the
        # resolver so the file is found in the new layout, the legacy flat
        # layout, $DOCUVERSE_HOME, or packaged defaults — in that order.
        # Falls back to the original string so existing absolute paths still
        # work.
        resolved = resolve_optional(self._strip_config_prefix(config)) or config
        servers_configs = read_config_file(resolved)
        self.servers = {}
        if servers_configs:
            for server_id, server_config in servers_configs.items():
                self.servers[server_id] = Server(**server_config)

    @staticmethod
    def _strip_config_prefix(path: str) -> str:
        """Resolver wants paths relative to a config base dir.

        Callers historically pass ``"config/<file>"``; strip the leading
        ``config/`` so the resolver can search its own tiers.
        """
        if path.startswith("config/"):
            return path[len("config/"):]
        if path.startswith("./config/"):
            return path[len("./config/"):]
        return path

    @classmethod
    def from_inline(cls, servers_dict: dict[str, dict]) -> "RetrievalServers":
        """Construct directly from an in-memory ``{name: {host, port, ...}}`` dict.

        Lets users carry server specs inline in a YAML config without a
        separate registry file::

            server:
              my-prod:
                host: milvus.example.com
                port: 19530
                secure: true
        """
        instance = cls.__new__(cls)
        instance.servers = {
            name: Server(**spec) for name, spec in servers_dict.items()
        }
        return instance

    def get(self, name: str, default=None):
        return self.servers.get(name, default)

    def __getitem__(self, item):
        return self.servers.get(item)

    def items(self):
        # Bug fix: previously returned ``self.servers.__dict__.items()`` which
        # iterates the dict's internals, not the (name, Server) pairs callers
        # expect.
        return self.servers.items()

