"""Preview or push this prepared Git repository to ChananyuKML on GitHub."""

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "ChananyuKML"


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT,
                                   stderr=subprocess.PIPE, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="aig-dec", help="Name of an existing empty GitHub repository")
    parser.add_argument("--push", action="store_true", help="Actually set the remote and push main; default only previews")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.repository) or args.repository in (".", ".."):
        parser.error("Invalid repository name.")
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise ValueError("Initialize Git inside this project first, following docs/GITHUB.md.")
    if git("branch", "--show-current") != "main":
        raise ValueError("This helper expects the prepared main branch.")
    git("rev-parse", "--verify", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("Commit or review the outstanding source changes before publishing.")
    files = git("ls-files").splitlines()
    blocked = []
    for name in files:
        path = Path(name)
        if (path.parts[0] in (".venv", "runs", "output", "checkpoints", "tmp", "local_archive")
                or (path.parts[0] == "dataset" and name != "dataset/README.md")
                or path.suffix.lower() in (".zip", ".pt", ".pth", ".pem", ".key")
                or path.name == ".env" or path.name.startswith(".env.")
                or (ROOT / name).stat().st_size > 10 * 1024 * 1024):
            blocked.append(name)
    if blocked:
        raise ValueError("Remove local artifacts from Git tracking before upload: " + ", ".join(blocked))
    url = f"https://github.com/{ACCOUNT}/{args.repository}.git"
    try:
        current_remote = git("remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        current_remote = None
    accepted = (url, url.removesuffix(".git"), f"git@github.com:{ACCOUNT}/{args.repository}.git")
    if current_remote and current_remote not in accepted:
        raise ValueError("Existing origin points elsewhere; inspect it before publishing. The helper will not replace it.")
    commands = ([] if current_remote else [["git", "remote", "add", "origin", url]])
    commands.append(["git", "push", "-u", "origin", "main"])
    print(f"Destination: {ACCOUNT}/{args.repository}\nReviewed source files: {len(files)}")
    for command in commands:
        print(subprocess.list2cmdline(command))
    if not args.push:
        print("Preview only. Create the empty GitHub repository, then repeat with --push.")
        return
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
