import re
import subprocess

def get_linux_distribution():
	try:
		try:
			with open("/etc/os-release", encoding="utf-8") as file_handle:
				os_release_content = file_handle.read()
			name_match = re.search(
			 '^(?:ID|NAME)="?([^"\\n]+)"?',
			 os_release_content,
			 flags=re.MULTILINE,
			)
			version_match = re.search(
			 '^VERSION_ID="?([^"\\n]+)"?',
			 os_release_content,
			 flags=re.MULTILINE,
			)
			if name_match and version_match:
				distro = name_match.group(1).strip().lower().split(" ")[0]
				release = version_match.group(1).strip()
				return (distro, release)
		except FileNotFoundError:
			pass
		try:
			result = subprocess.run(
			 ["lsb_release", "-a"],
			 capture_output=True,
			 text=True,
			 check=True,
			)
			output = result.stdout
			distro_match = re.search("Distributor ID:\\s*([^\\n]+)", output)
			release_match = re.search("Release:\\s*([^\\n]+)", output)
			if distro_match and release_match:
				distro = distro_match.group(1).strip().lower().split(" ")[0]
				release = release_match.group(1).strip()
				return (distro, release)
		except (FileNotFoundError, subprocess.CalledProcessError):
			pass
		return (None, None)
	except Exception:
		return (None, None)
