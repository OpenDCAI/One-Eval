"""
外部仓库评测执行器
负责运行 external_repo 类型的 benchmark

修复了 3 个阻塞级问题：
1. 仓库按 URL/ref 隔离，用 .setup_complete 标记安装状态
2. 使用 argv + shell=False，避免命令注入和密钥泄露
3. 为每个仓库创建独立 venv，不污染 One-Eval 环境

Docker 沙箱支持：
当 repo_eval.env_requires.sandbox == "docker" 时，评测命令在容器内执行，
隔离生成代码的执行环境，防止恶意代码影响宿主机。
"""
import os
import subprocess
import json
import csv
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from glob import glob

from one_eval.logger import get_logger

log = get_logger("ExternalRepoRunner")


class ExternalRepoRunner:
    """运行外部仓库的评测工具"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化 ExternalRepoRunner

        Args:
            cache_dir: 外部仓库的缓存目录
                      默认使用项目根目录的 cache/external_repos
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            # 默认使用项目根目录的 cache/external_repos
            project_root = Path(__file__).resolve().parents[2]
            self.cache_dir = project_root / "cache" / "external_repos"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"ExternalRepoRunner 初始化，缓存目录: {self.cache_dir}")

    def _sanitize_ref(self, ref: str) -> str:
        """
        清理 ref 名称，使其可以作为目录名

        Args:
            ref: git ref (tag/branch/commit)

        Returns:
            安全的目录名
        """
        # 替换不安全的字符
        safe_ref = ref.replace("/", "_").replace("\\", "_")
        return safe_ref

    def _get_repo_root(self, repo_url: str, repo_ref: str) -> Path:
        """
        获取仓库的根目录路径

        目录结构: cache/external_repos/{repo_name}/{ref}/

        Args:
            repo_url: 仓库 URL
            repo_ref: git ref

        Returns:
            仓库根目录路径
        """
        repo_name = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        safe_ref = self._sanitize_ref(repo_ref)
        repo_root = self.cache_dir / repo_name / safe_ref

        return repo_root

    def _get_venv_python(self, venv_path: Path) -> Path:
        """
        获取虚拟环境中的 Python 可执行文件路径

        Args:
            venv_path: 虚拟环境路径

        Returns:
            Python 可执行文件路径
        """
        if os.name == "nt":  # Windows
            return venv_path / "Scripts" / "python.exe"
        else:  # Unix-like
            return venv_path / "bin" / "python"

    def _verify_repo(
        self,
        repo_path: Path,
        repo_url: str,
        repo_ref: str
    ) -> Dict[str, Any]:
        """
        验证已存在的仓库是否匹配配置

        Args:
            repo_path: 仓库路径
            repo_url: 期望的仓库 URL
            repo_ref: 期望的 ref

        Returns:
            验证结果
        """
        try:
            # 检查 remote URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            actual_url = result.stdout.strip()

            # 标准化 URL 比较（移除 .git 后缀）
            expected_url = repo_url.rstrip("/").removesuffix(".git")
            actual_url = actual_url.rstrip("/").removesuffix(".git")

            if expected_url != actual_url:
                return {
                    "ok": False,
                    "stage": "verify",
                    "error": f"仓库 URL 不匹配。期望: {expected_url}, 实际: {actual_url}"
                }

            # 检查当前 ref
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            current_commit = result.stdout.strip()

            # 获取期望的 commit
            result = subprocess.run(
                ["git", "rev-parse", repo_ref],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            expected_commit = result.stdout.strip()

            if current_commit != expected_commit:
                return {
                    "ok": False,
                    "stage": "verify",
                    "error": f"Git ref 不匹配。期望: {repo_ref} ({expected_commit[:8]}), 实际: {current_commit[:8]}"
                }

            return {"ok": True}

        except subprocess.CalledProcessError as e:
            return {
                "ok": False,
                "stage": "verify",
                "error": f"验证仓库失败: {e.stderr}"
            }

    def setup_repo(self, repo_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        设置外部仓库（clone + venv + install）

        Args:
            repo_config: repo_eval 配置，包含 repo_url, ref, setup 等

        Returns:
            {
                "ok": bool,
                "repo_path": str,  # 仓库路径
                "venv_path": str,  # 虚拟环境路径
                "python": str,     # Python 可执行文件路径
                "stage": str,      # 失败时的阶段
                "error": str       # 失败时的错误信息
            }
        """
        repo_url = repo_config.get("repo_url")
        repo_ref = repo_config.get("ref", "main")

        if not repo_url:
            return {"ok": False, "stage": "config", "error": "repo_url 未配置"}

        # 获取目录结构
        repo_root = self._get_repo_root(repo_url, repo_ref)
        repo_path = repo_root / "repo"
        venv_path = repo_root / ".venv"
        setup_complete_marker = repo_root / ".setup_complete"

        # 如果已经设置完成，验证后直接返回
        if setup_complete_marker.exists():
            log.info(f"检测到已完成的安装: {repo_root}")

            # 验证仓库状态
            verify_result = self._verify_repo(repo_path, repo_url, repo_ref)
            if not verify_result["ok"]:
                log.warning(f"仓库验证失败: {verify_result['error']}")
                return verify_result

            python_path = self._get_venv_python(venv_path)
            if not python_path.exists():
                return {
                    "ok": False,
                    "stage": "verify",
                    "error": f"虚拟环境 Python 不存在: {python_path}"
                }

            # 确保 patch 文件存在（同一仓库可能被多个 bench 共用，各自有不同的 patch）
            project_root = Path(__file__).resolve().parents[2]
            patch_files = repo_config.get("patch_files", [])
            for patch in patch_files:
                src = project_root / patch["src"]
                dest = repo_path / patch["dest"]
                if src.exists():
                    if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest)
                        log.info(f"✅ 更新 patch: {patch['src']} -> {patch['dest']}")

            log.info("✅ 仓库验证通过，跳过安装")
            return {
                "ok": True,
                "repo_path": str(repo_path),
                "venv_path": str(venv_path),
                "python": str(python_path)
            }

        # 如果仓库目录存在但未完成安装，返回错误
        if repo_root.exists():
            return {
                "ok": False,
                "stage": "setup",
                "error": f"仓库目录存在但安装未完成: {repo_root}\n请删除该目录后重试"
            }

        # 创建目录结构
        repo_root.mkdir(parents=True, exist_ok=True)

        # Clone 仓库
        try:
            log.info(f"克隆仓库: {repo_url} -> {repo_path}")
            subprocess.run(
                ["git", "clone", repo_url, str(repo_path)],
                check=True,
                capture_output=True,
                text=True
            )
            log.info("✅ 克隆完成")

        except subprocess.CalledProcessError as e:
            shutil.rmtree(repo_root, ignore_errors=True)
            return {
                "ok": False,
                "stage": "clone",
                "error": f"克隆仓库失败: {e.stderr}"
            }

        # Checkout 到指定 ref
        try:
            log.info(f"切换到 ref: {repo_ref}")
            subprocess.run(
                ["git", "checkout", repo_ref],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            log.info("✅ Checkout 完成")

        except subprocess.CalledProcessError as e:
            shutil.rmtree(repo_root, ignore_errors=True)
            return {
                "ok": False,
                "stage": "checkout",
                "error": f"切换 ref 失败: {e.stderr}"
            }

        # 创建虚拟环境
        try:
            log.info(f"创建虚拟环境: {venv_path}")

            # 解析 Python 版本要求
            env_requires = repo_config.get("env_requires", {})
            python_version = None

            if isinstance(env_requires, dict):
                python_req = env_requires.get("python", "")
                if python_req:
                    # 提取版本号，支持 ">=3.12", "3.12", ">=3.12,<4.0" 等格式
                    import re
                    match = re.search(r'(\d+\.\d+)', python_req)
                    if match:
                        python_version = match.group(1)
                        log.info(f"检测到 Python 版本要求: {python_req}，将使用 Python {python_version}")

            # 构建 uv venv 命令
            venv_cmd = ["uv", "venv", str(venv_path)]
            if python_version:
                venv_cmd.extend(["--python", python_version])

            subprocess.run(
                venv_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            log.info("✅ 虚拟环境创建完成")

        except subprocess.CalledProcessError as e:
            shutil.rmtree(repo_root, ignore_errors=True)
            error_msg = f"创建虚拟环境失败: {e.stderr}"

            # 如果是 Python 版本找不到的错误，给出友好提示
            if python_version and ("not found" in e.stderr.lower() or "no python" in e.stderr.lower()):
                error_msg += (
                    f"\n\n💡 提示：此 benchmark 需要 Python {python_version}。"
                    f"\n   请确保系统已安装 Python {python_version}，并且 'uv' 能找到它。"
                    f"\n   Windows 用户可运行: py -{python_version} --version 检查"
                    f"\n   安装方式: winget install Python.Python.{python_version.replace('.', '')}"
                )

            return {
                "ok": False,
                "stage": "venv",
                "error": error_msg
            }

        # 获取 Python 路径
        python_path = self._get_venv_python(venv_path)
        if not python_path.exists():
            shutil.rmtree(repo_root, ignore_errors=True)
            return {
                "ok": False,
                "stage": "venv",
                "error": f"虚拟环境 Python 不存在: {python_path}"
            }

        # 复制 patch 文件（在 setup 命令之前执行）
        project_root = Path(__file__).resolve().parents[2]
        patch_files = repo_config.get("patch_files", [])
        for patch in patch_files:
            src = project_root / patch["src"]
            dest = repo_path / patch["dest"]
            if not src.exists():
                shutil.rmtree(repo_root, ignore_errors=True)
                return {
                    "ok": False,
                    "stage": "patch",
                    "error": f"Patch 文件不存在: {src}"
                }
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            log.info(f"✅ 复制 patch: {patch['src']} -> {patch['dest']}")

        # 运行安装命令（docker_skip_setup 时跳过宿主机安装）
        setup_commands = repo_config.get("setup", [])
        if self._should_use_docker(repo_config) and repo_config.get("docker_skip_setup", False):
            setup_commands = []
            log.info("⏭️ Docker 镜像已预装依赖，跳过宿主机 setup 命令")
        project_root = Path(__file__).resolve().parents[2]
        for i, cmd_template in enumerate(setup_commands, 1):
            # 替换占位符
            variables = {
                "{{venv_path}}": str(venv_path),
                "{{repo_dir}}": str(repo_path),
                "{{python}}": str(python_path),
                "{{uv}}": "uv",
                "{{oneeval_root}}": str(project_root)
            }

            # 解析命令（支持字符串，转换为 argv）
            if isinstance(cmd_template, str):
                for placeholder, value in variables.items():
                    cmd_template = cmd_template.replace(placeholder, value)
                argv = cmd_template.split()
            else:
                argv = []
                for arg in cmd_template:
                    for placeholder, value in variables.items():
                        arg = arg.replace(placeholder, value)
                    argv.append(arg)

            try:
                log.info(f"执行安装命令 [{i}/{len(setup_commands)}]: {' '.join(argv[:3])}...")
                subprocess.run(
                    argv,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=repo_path,
                    encoding="utf-8",
                    errors="replace"
                )
                log.info(f"✅ 安装命令 {i} 完成")

            except subprocess.CalledProcessError as e:
                shutil.rmtree(repo_root, ignore_errors=True)
                return {
                    "ok": False,
                    "stage": "install",
                    "error": f"安装命令失败: {' '.join(argv)}\n{e.stderr}"
                }

        # 标记安装完成
        setup_complete_marker.write_text(
            json.dumps({
                "repo_url": repo_url,
                "ref": repo_ref,
                "completed_at": str(Path(__file__).stat().st_mtime)
            })
        )

        log.info("✅ 仓库设置完成")
        return {
            "ok": True,
            "repo_path": str(repo_path),
            "venv_path": str(venv_path),
            "python": str(python_path)
        }

    def _should_use_docker(self, repo_config: Dict[str, Any]) -> bool:
        """判断是否需要使用 Docker 沙箱执行"""
        env_requires = repo_config.get("env_requires", {})
        if isinstance(env_requires, dict):
            sandbox = env_requires.get("sandbox", "none")
        elif isinstance(env_requires, list):
            return False
        else:
            return False
        return sandbox == "docker"

    def _check_docker_available(self) -> Dict[str, Any]:
        """检查 Docker 是否可用"""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return {
                    "ok": False,
                    "error": "Docker daemon 未运行或无权限访问"
                }
            return {"ok": True}
        except FileNotFoundError:
            return {"ok": False, "error": "Docker 未安装"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Docker info 超时"}

    def _ensure_docker_image(self, repo_config: Dict[str, Any], repo_path: Path) -> Dict[str, Any]:
        """
        确保预构建的 Docker 镜像存在。如果不存在则自动构建。

        构建流程：
        1. 查找 bench 对应的 Dockerfile（在 patches 目录中）
        2. 将仓库源码作为 build context 的一部分
        3. docker build 生成镜像

        Returns:
            {"ok": True, "image": str} 或 {"ok": False, "error": str}
        """
        docker_image = repo_config.get("docker_image", "python:3.11-slim")

        # 检查镜像是否已存在
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", docker_image],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                log.info(f"✅ Docker 镜像已存在: {docker_image}")
                return {"ok": True, "image": docker_image}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # 镜像不存在，尝试构建
        dockerfile_path = repo_config.get("dockerfile")
        if not dockerfile_path:
            return {"ok": True, "image": docker_image}

        project_root = Path(__file__).resolve().parents[2]
        dockerfile_full = project_root / dockerfile_path
        if not dockerfile_full.exists():
            return {"ok": False, "error": f"Dockerfile 不存在: {dockerfile_full}"}

        # 构建 context：Dockerfile 所在目录 + 仓库源码
        # 创建临时 build context（软链接仓库到 Dockerfile 旁边的 repo/）
        build_context = dockerfile_full.parent
        context_repo_link = build_context / "repo"

        # 如果已存在则删除旧链接
        if context_repo_link.exists() or context_repo_link.is_symlink():
            if context_repo_link.is_symlink() or context_repo_link.is_file():
                context_repo_link.unlink()
            else:
                shutil.rmtree(context_repo_link)

        # 在 Windows 上 symlink 可能需要管理员权限，改用 junction 或直接复制
        # 这里用复制 pyproject.toml + lcb_runner 目录（只需安装用）
        context_repo_link.mkdir(parents=True, exist_ok=True)
        repo_src = repo_path / "repo" if (repo_path / "repo").exists() else repo_path

        # 只复制安装所需的文件
        for item in ["pyproject.toml", "setup.py", "setup.cfg", "lcb_runner", "README.md"]:
            src_item = repo_src / item
            dst_item = context_repo_link / item
            if src_item.exists():
                if src_item.is_dir():
                    shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_item, dst_item)

        log.info(f"🐳 构建 Docker 镜像: {docker_image} (首次需要 10-20 分钟)")
        try:
            result = subprocess.run(
                ["docker", "build", "-t", docker_image, "-f", str(dockerfile_full), str(build_context)],
                capture_output=True, text=True, timeout=3600
            )
            if result.returncode != 0:
                log.error(f"Docker build 失败:\n{result.stderr[-2000:]}")
                return {"ok": False, "error": f"Docker build 失败: {result.stderr[-500:]}"}
            log.info(f"✅ Docker 镜像构建成功: {docker_image}")
            return {"ok": True, "image": docker_image}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Docker build 超时 (>1h)"}
        finally:
            # 清理 build context 中的 repo 副本
            if context_repo_link.exists():
                shutil.rmtree(context_repo_link, ignore_errors=True)

    def _build_docker_argv(
        self,
        argv: List[str],
        repo_path: Path,
        output_dir_path: Path,
        work_dir: Path,
        env_vars: Dict[str, str],
        repo_config: Dict[str, Any],
        venv_path: Optional[Path] = None,
    ) -> List[str]:
        """
        构建 docker run 命令，将评测包装在容器中执行。

        容器策略：
        - 挂载仓库目录（rw，用于 pip install -e .）
        - 挂载输出目录（rw）
        - 通过 --env 传入 API 密钥等环境变量（不落盘）
        - 设置内存和 CPU 限制防止资源耗尽
        - --cap-drop ALL 最小权限
        - 容器内自动安装依赖（pip install -e .）后再执行评测
        """
        docker_image = repo_config.get("docker_image", "python:3.11-slim")

        container_repo = "/workspace/repo"
        container_output = "/workspace/output"

        # 缓存目录：pip/uv 包缓存 + HuggingFace 数据集缓存
        cache_root = self.cache_dir.parent / "docker_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        pip_cache = cache_root / "pip"
        uv_cache = cache_root / "uv"
        hf_cache = cache_root / "huggingface"
        pip_cache.mkdir(exist_ok=True)
        uv_cache.mkdir(exist_ok=True)
        hf_cache.mkdir(exist_ok=True)

        docker_argv = [
            "docker", "run",
            "--rm",
            "--name", f"oneeval-{repo_path.name}-{os.getpid()}",
            "--memory", "16g",
            "--cpus", "4",
            "--cap-drop", "ALL",
            "--cap-add", "NET_RAW",
            "--security-opt", "no-new-privileges",
            "-v", f"{repo_path}:{container_repo}:rw",
            "-v", f"{output_dir_path}:{container_output}:rw",
            # 挂载缓存目录（持久化 pip/uv 包 + HF 数据集）
            "-v", f"{pip_cache}:/root/.cache/pip:rw",
            "-v", f"{uv_cache}:/root/.cache/uv:rw",
            "-v", f"{hf_cache}:/root/.cache/huggingface:rw",
            "--env", "HF_HOME=/root/.cache/huggingface",
            "--env", f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')}",
            "--env", "UV_LINK_MODE=copy",
            "-w", container_repo,
        ]

        for key, value in env_vars.items():
            docker_argv.extend(["--env", f"{key}={value}"])

        # 代理环境变量：将 localhost/127.0.0.1 替换为 host.docker.internal
        for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            val = os.environ.get(proxy_var)
            if val:
                # Docker 容器内访问宿主机需要用 host.docker.internal
                val = val.replace("localhost", "host.docker.internal")
                val = val.replace("127.0.0.1", "host.docker.internal")
                docker_argv.extend(["--env", f"{proxy_var}={val}"])

        docker_argv.append(docker_image)

        # 如果镜像已预装依赖，跳过 setup 步骤直接执行评测
        skip_setup = repo_config.get("docker_skip_setup", False)

        if skip_setup:
            install_cmd = ""
        else:
            # 容器内命令：先安装 uv，然后按 setup 中定义的命令安装依赖
            setup_commands = repo_config.get("setup", [])
            install_parts = ["pip install uv"]
            if setup_commands:
                for cmd in setup_commands:
                    resolved = cmd.replace("{{uv}}", "uv").replace("{{repo_dir}}", container_repo)
                    if "uv pip install" in resolved and "--system" not in resolved:
                        resolved = resolved.replace("uv pip install", "uv pip install --system")
                    install_parts.append(resolved)
            else:
                install_parts.append(f"uv pip install --system -e {container_repo}")

            install_cmd = " && ".join(install_parts)

        container_argv = []
        for arg in argv:
            replaced = arg
            replaced = replaced.replace(str(repo_path), container_repo)
            replaced = replaced.replace(str(output_dir_path), container_output)
            if replaced.endswith("python") or replaced.endswith("python.exe") or "Scripts" in replaced or "/bin/python" in replaced:
                replaced = "python"
            container_argv.append(replaced)

        eval_cmd = " ".join(container_argv)
        combined_cmd = f"{install_cmd} && {eval_cmd}" if install_cmd else eval_cmd

        docker_argv.extend(["bash", "-c", combined_cmd])
        return docker_argv

    def run_evaluation(
        self,
        repo_config: Dict[str, Any],
        repo_path: str,
        python_path: str,
        env_vars: Dict[str, str],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        运行外部评测

        当 repo_config.env_requires.sandbox == "docker" 时，
        评测命令在 Docker 容器内执行，隔离代码执行环境。

        Args:
            repo_config: repo_eval 配置
            repo_path: 仓库路径
            python_path: 虚拟环境中的 Python 路径
            env_vars: 环境变量（如 OPENAI_API_KEY, OPENAI_API_BASE）
            output_dir: 输出目录

        Returns:
            {
                "ok": bool,
                "output_dir": str,
                "log_path": str,
                "returncode": int,
                "stage": str,
                "error": str
            }
        """
        run_config = repo_config.get("run", {})
        argv_template = run_config.get("argv")
        work_dir_template = run_config.get("work_dir", "{{repo_dir}}")

        if not argv_template:
            return {
                "ok": False,
                "stage": "config",
                "error": "run.argv 未配置"
            }

        # 转换为绝对路径
        repo_path = Path(repo_path).resolve()
        output_dir_path = Path(output_dir).resolve()
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # 准备变量替换
        variables = {
            "{{repo_dir}}": str(repo_path),
            "{{output_dir}}": str(output_dir_path),
            "{{python}}": python_path
        }

        # 替换 work_dir
        work_dir = work_dir_template
        for placeholder, value in variables.items():
            work_dir = work_dir.replace(placeholder, value)
        work_dir = Path(work_dir).resolve()

        # 构建 argv（不替换环境变量）
        argv = []
        for arg in argv_template:
            replaced = arg
            for placeholder, value in variables.items():
                replaced = replaced.replace(placeholder, value)
            argv.append(replaced)

        # 设置环境变量
        env = os.environ.copy()
        env.update(env_vars)

        # 添加网络容错配置
        for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            if proxy_var not in env and proxy_var in os.environ:
                env[proxy_var] = os.environ[proxy_var]

        if os.getenv("DISABLE_SSL_VERIFY") == "1":
            env["CURL_CA_BUNDLE"] = ""
            env["REQUESTS_CA_BUNDLE"] = ""
            log.warning("SSL 验证已禁用（DISABLE_SSL_VERIFY=1）")

        # 准备日志文件
        log_path = output_dir_path / "external_repo.log"

        # 判断是否使用 Docker 沙箱
        use_docker = self._should_use_docker(repo_config)
        if use_docker:
            docker_check = self._check_docker_available()
            if not docker_check["ok"]:
                return {
                    "ok": False,
                    "stage": "docker",
                    "error": f"需要 Docker 沙箱但不可用: {docker_check['error']}"
                }
            # 预构建镜像：如果配置了 dockerfile，确保镜像存在
            if repo_config.get("dockerfile"):
                image_check = self._ensure_docker_image(repo_config, repo_path)
                if not image_check["ok"]:
                    return {
                        "ok": False,
                        "stage": "docker_build",
                        "error": image_check["error"]
                    }
            run_argv = self._build_docker_argv(
                argv, repo_path, output_dir_path, work_dir, env_vars, repo_config
            )
            log.info("🐳 使用 Docker 沙箱执行评测")
        else:
            run_argv = argv

        # 运行评测
        try:
            log.info(f"运行评测: {run_argv[0]} {run_argv[1] if len(run_argv) > 1 else ''} ...")
            log.info(f"工作目录: {work_dir}")
            log.info(f"输出目录: {output_dir_path}")
            log.info(f"日志文件: {log_path}")

            with log_path.open("w", encoding="utf-8") as log_file:
                if use_docker:
                    result = subprocess.run(
                        run_argv,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=14400
                    )
                else:
                    result = subprocess.run(
                        run_argv,
                        cwd=work_dir,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=7200
                    )

            if result.returncode != 0:
                # 读取日志尾部
                log_tail = log_path.read_text(encoding="utf-8")[-2000:]

                # 分析错误类型并提供建议
                error_type = "unknown"
                suggestion = ""

                if "SSL" in log_tail or "ssl" in log_tail.lower():
                    error_type = "network_ssl"
                    suggestion = (
                        "网络 SSL 错误。建议：\n"
                        "1. 检查网络连接\n"
                        "2. 配置 HTTP_PROXY 和 HTTPS_PROXY 环境变量\n"
                        "3. 或设置 DISABLE_SSL_VERIFY=1（仅测试用）"
                    )
                elif "download" in log_tail.lower() or "urlopen" in log_tail or "Failed to download" in log_tail:
                    error_type = "network_download"
                    suggestion = (
                        "数据下载失败。建议：\n"
                        "1. 检查网络连接\n"
                        "2. 配置代理（HTTP_PROXY, HTTPS_PROXY）\n"
                        "3. 手动预下载数据到仓库缓存目录"
                    )

                return {
                    "ok": False,
                    "stage": "evaluation",
                    "error": f"评测失败 (返回码: {result.returncode})",
                    "error_type": error_type,
                    "suggestion": suggestion,
                    "returncode": result.returncode,
                    "log_path": str(log_path),
                    "log_tail": log_tail,
                    "retryable": error_type in ["network_ssl", "network_download"]
                }

            # 即使返回码是0，也检查日志中是否有关键错误
            log_content = log_path.read_text(encoding="utf-8")

            # 检查是否有 "skipping this combination" 或类似的跳过信息
            if "skipping this combination" in log_content.lower() or "combination failed" in log_content.lower():
                log_tail = log_content[-2000:]
                error_type = "unknown"
                suggestion = ""

                if "SSL" in log_tail or "ssl" in log_tail.lower():
                    error_type = "network_ssl"
                    suggestion = (
                        "网络 SSL 错误导致数据下载失败。建议：\n"
                        "1. 检查网络连接\n"
                        "2. 配置 HTTP_PROXY 和 HTTPS_PROXY 环境变量\n"
                        "3. 或设置 DISABLE_SSL_VERIFY=1（仅测试用）"
                    )
                elif "download" in log_tail.lower() or "Failed to download" in log_tail:
                    error_type = "network_download"
                    suggestion = (
                        "数据下载失败。建议：\n"
                        "1. 检查网络连接\n"
                        "2. 配置代理（HTTP_PROXY, HTTPS_PROXY）\n"
                        "3. 手动预下载数据到仓库缓存目录"
                    )

                return {
                    "ok": False,
                    "stage": "evaluation",
                    "error": "评测被跳过（数据下载或其他前置步骤失败）",
                    "error_type": error_type,
                    "suggestion": suggestion,
                    "returncode": result.returncode,
                    "log_path": str(log_path),
                    "log_tail": log_tail,
                    "retryable": error_type in ["network_ssl", "network_download"]
                }

            log.info("✅ 评测完成")
            return {
                "ok": True,
                "output_dir": str(output_dir_path),
                "log_path": str(log_path),
                "returncode": 0
            }

        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "stage": "evaluation",
                "error": "评测超时（超过 1 小时）",
                "log_path": str(log_path)
            }
        except Exception as e:
            return {
                "ok": False,
                "stage": "evaluation",
                "error": f"运行失败: {str(e)}",
                "log_path": str(log_path) if log_path.exists() else None
            }

    def parse_results(
        self,
        result_config: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        解析评测结果

        Args:
            result_config: result 配置
            output_dir: 输出目录

        Returns:
            {
                "ok": bool,
                "score": float,
                "metric_name": str,
                "result_file": str,
                "stage": str,
                "error": str
            }
        """
        result_path_pattern = result_config.get("path", "")
        result_format = result_config.get("format", "json")
        score_path = result_config.get("score_path", "")
        metric_name = result_config.get("metric_name", "score")

        if not result_path_pattern:
            return {
                "ok": False,
                "stage": "config",
                "error": "result.path 未配置"
            }

        # 转换为绝对路径
        output_dir = Path(output_dir).resolve()

        # 查找结果文件
        full_pattern = str(output_dir / result_path_pattern)
        matched_files = [
            Path(path)
            for path in glob(full_pattern, recursive=True)
            if Path(path).is_file()
        ]

        if not matched_files:
            return {
                "ok": False,
                "stage": "parse",
                "error": f"未找到结果文件: {full_pattern}"
            }

        if len(matched_files) > 1:
            return {
                "ok": False,
                "stage": "parse",
                "error": f"匹配到多个结果文件 ({len(matched_files)} 个)，无法确定使用哪个",
                "matched_files": [str(p) for p in matched_files]
            }

        result_file = matched_files[0]
        log.info(f"找到结果文件: {result_file}")

        # 解析结果文件
        try:
            if result_format == "json":
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif result_format == "csv":
                with open(result_file, 'r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
            else:
                return {
                    "ok": False,
                    "stage": "parse",
                    "error": f"不支持的格式: {result_format}"
                }

            # 提取分数
            score = self._extract_score(data, score_path, result_format)

            if score is None:
                return {
                    "ok": False,
                    "stage": "parse",
                    "error": f"无法从结果中提取分数 (score_path: {score_path})"
                }

            log.info(f"✅ 解析完成，{metric_name}: {score}")
            parsed = {
                "ok": True,
                "score": float(score),
                "metric_name": metric_name,
                "result_file": str(result_file),
            }
            # Pass through additional fields from the scores JSON for report rendering
            if result_format == "json" and isinstance(data, dict):
                if data.get("total_samples") is not None:
                    parsed["total_samples"] = int(data["total_samples"])
                if data.get("detail_path"):
                    parsed["detail_path"] = data["detail_path"]
            return parsed

        except Exception as e:
            return {
                "ok": False,
                "stage": "parse",
                "error": f"解析结果失败: {str(e)}"
            }

    def _extract_score(
        self,
        data: Any,
        score_path: str,
        result_format: str
    ) -> Optional[float]:
        """
        从结果数据中提取分数

        Args:
            data: 结果数据（dict 或 list）
            score_path: 分数路径
            result_format: 结果格式

        Returns:
            分数值（float）或 None
        """
        if not score_path:
            return None

        # CSV: 从第一行提取
        if result_format == "csv" and isinstance(data, list):
            if len(data) == 0:
                return None
            row = data[0]
            if score_path in row:
                try:
                    return float(row[score_path])
                except (ValueError, TypeError):
                    return None
            return None

        # JSON: 使用点路径
        if isinstance(data, dict):
            keys = score_path.split(".")
            current = data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            try:
                return float(current) if current is not None else None
            except (ValueError, TypeError):
                return None

        return None
