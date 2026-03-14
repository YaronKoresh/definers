import importlib
import os
import site
import subprocess
import sys

from definers.platform.paths import (
 cwd,
 full_path,
 normalize_path,
 parent_directory,
 tmp,
 unique,
)
from definers.platform.services import get_infrastructure_services


def add_path(*p):
	joined = os.path.join(*[str(part).strip() for part in p]) if p else ""
	path = joined if joined == "" else os.path.abspath(os.path.expanduser(joined))
	normalized_path = path if path == "" else os.path.normcase(path)
	normalized_sys_path = {
	 entry if entry == "" else os.path.normcase(entry) for entry in sys.path
	}
	infrastructure = get_infrastructure_services()
	filesystem = infrastructure.filesystem
	environment = infrastructure.environment
	processes = infrastructure.processes
	if normalized_path not in normalized_sys_path:
		filesystem.permit(
		 path,
		 exists_func=filesystem.exist,
		 get_os_name_func=environment.get_os_name,
		 subprocess_module=subprocess,
		)
		sys.path.insert(0, path)
		site.addsitedir(path)
		importlib.invalidate_caches()
	os_name = environment.get_os_name()
	if os_name == "linux" or os_name == "darwin":
		command = f'export PATH="{path}:$PATH"'
		shell_config_path = None
		if filesystem.exist("~/.bashrc"):
			shell_config_path = "~/.bashrc"
		elif filesystem.exist("~/.zshrc"):
			shell_config_path = "~/.zshrc"
		if shell_config_path is not None:
			content = filesystem.read(shell_config_path)
			if content is not None:
				filesystem.write(shell_config_path, "\n".join([content, command]))
		return processes.run(command)
	if os_name == "windows":
		return processes.run(f'setx PATH "%PATH%;{path}"')
	return None


def find_package_paths(package_name):
	package_paths_found = []
	package_dir_name = package_name.replace("-", "_")
	site_packages_dirs = site.getsitepackages()
	for site_packages_dir in site_packages_dirs:
		package_path = os.path.join(site_packages_dir, package_dir_name)
		if os.path.exists(package_path) and os.path.isdir(package_path):
			package_paths_found.append(package_path)
	for search_path in sys.path:
		if search_path:
			potential_package_path = os.path.join(search_path, package_dir_name)
			if os.path.exists(potential_package_path) and os.path.isdir(
			 potential_package_path
			):
				package_paths_found.append(potential_package_path)
	for site_packages_dir in site_packages_dirs:
		dist_packages_dir = site_packages_dir.replace(
		 "site-packages", "dist-packages"
		)
		if dist_packages_dir != site_packages_dir:
			package_path = os.path.join(dist_packages_dir, package_dir_name)
			if os.path.exists(package_path) and os.path.isdir(package_path):
				package_paths_found.append(package_path)
	return unique(package_paths_found)


def is_package_path(package_path, package_name=None):
	if not package_path:
		return False
	if not os.path.isdir(package_path):
		return False
	has_package_markers = (
	 os.path.exists(os.path.join(package_path, "__init__.py"))
	 or os.path.exists(
	  os.path.join(package_path, os.path.basename(package_path))
	 )
	 or os.path.exists(os.path.join(package_path, "src"))
	)
	if not has_package_markers:
		return False
	if package_name is None:
		return True
	return package_name == os.path.basename(package_path)
