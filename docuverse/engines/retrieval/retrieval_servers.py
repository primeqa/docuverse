import json
import os
from dataclasses import dataclass, field
from typing import Optional

# from jupyter_lsp.specs import yaml
import yaml

from docuverse.utils import get_config_dir
from docuverse.utils import read_config_file

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
        # config = os.path.join(get_config_dir(os.path.dirname(config)), os.path.basename(config))
        servers_configs = read_config_file(config)
        self.servers = {}
        for server_id, server_config in servers_configs.items():
            self.servers[server_id] = Server(**server_config)

    def get(self, name: str, default=None):
        return self.servers.get(name, None)

    def __getitem__(self, item):
        return self.servers.get(item)

    def items(self):
        return self.servers.__dict__.items()

