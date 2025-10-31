"""
CLI 主程式

命令列介面的入口點。
"""

import click
import json


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """OmniFoundry - 開源模型推論環境集成程式"""
    pass


@cli.command()
def list():
    """列出可用模型"""
    click.echo("此功能尚未實作")


@cli.command()
@click.argument("model_id")
def download(model_id):
    """下載模型"""
    click.echo(f"下載模型: {model_id}")
    click.echo("此功能尚未實作")


@cli.command()
@click.argument("model_id")
@click.option("--prompt", "-p", help="推論提示詞")
def run(model_id, prompt):
    """執行推論"""
    click.echo(f"執行模型: {model_id}")
    if prompt:
        click.echo(f"提示詞: {prompt}")
    click.echo("此功能尚未實作")


@cli.command()
@click.argument("model_id")
@click.option("--host", default="0.0.0.0", help="API 伺服器主機")
@click.option("--port", default=8000, help="API 伺服器埠號")
def serve(model_id, host, port):
    """啟動 API 服務"""
    click.echo(f"啟動 API 服務")
    click.echo(f"模型: {model_id}")
    click.echo(f"位址: http://{host}:{port}")
    click.echo("此功能尚未實作")


@cli.command()
@click.option("--json-output", "-j", is_flag=True, help="以 JSON 格式輸出")
@click.option("--recommend", "-r", help="推薦配置（格式: model_type:model_size，例如: llm:7b）")
def info(json_output, recommend):
    """顯示硬體資訊"""
    from omnifoundry.core.hardware import HardwareDetector
    
    detector = HardwareDetector()
    
    # 如果需要推薦配置
    if recommend:
        try:
            parts = recommend.split(":")
            if len(parts) == 2:
                model_type, model_size = parts
                config = detector.recommend_config(model_type=model_type, model_size=model_size)
                
                if json_output:
                    click.echo(json.dumps(config, indent=2, ensure_ascii=False))
                else:
                    click.echo(f"\n推薦配置（{model_type} - {model_size}）:")
                    click.echo(f"  裝置: {config['device']}")
                    click.echo(f"  資料型別: {config['dtype']}")
                    click.echo(f"  量化: {config['quantization'] or '無'}")
                    click.echo(f"  批次大小: {config['batch_size']}")
                    click.echo(f"  最大長度: {config.get('max_length', 'N/A')}")
                    click.echo(f"  Flash Attention: {'是' if config.get('use_flash_attention') else '否'}")
                    click.echo(f"  CPU Offload: {'是' if config.get('cpu_offload') else '否'}")
            else:
                click.echo("錯誤: 推薦配置格式應為 model_type:model_size", err=True)
                click.echo("範例: --recommend llm:7b", err=True)
        except Exception as e:
            click.echo(f"錯誤: {e}", err=True)
        return
    
    # 顯示硬體資訊
    if json_output:
        hw_info = detector.get_hardware_info()
        click.echo(json.dumps(hw_info, indent=2, ensure_ascii=False))
    else:
        detector.print_hardware_info()


if __name__ == "__main__":
    cli()

