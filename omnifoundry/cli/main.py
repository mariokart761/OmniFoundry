"""
CLI 主程式

命令列介面的入口點。
"""

import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """OmniFoundry - 開源模型推論環境集成程式"""
    pass


@cli.command()
def list():
    """列出可用模型"""
    pass


@cli.command()
@click.argument("model_id")
def download(model_id):
    """下載模型"""
    pass


@cli.command()
@click.argument("model_id")
@click.option("--prompt", "-p", help="推論提示詞")
def run(model_id, prompt):
    """執行推論"""
    pass


@cli.command()
@click.argument("model_id")
@click.option("--host", default="0.0.0.0", help="API 伺服器主機")
@click.option("--port", default=8000, help="API 伺服器埠號")
def serve(model_id, host, port):
    """啟動 API 服務"""
    pass


@cli.command()
def info():
    """顯示硬體資訊"""
    pass


if __name__ == "__main__":
    cli()

