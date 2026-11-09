"""Create (or update) the team's Hugging Face pieces. Safe to re-run.

    .venv/bin/python deploy/setup_hf.py <org>

  <org>/intake       private dataset: job queue, uploads, results, CLAUDE.md + skills
  <org>/intake-app   private static Space: the upload / results web page

Needs `hf auth login` with write access to <org>. Static Spaces are free for
organizations; the page uses each visitor's own HF sign-in, so no token is
stored in the Space.
"""
import pathlib, sys, tempfile, shutil
from huggingface_hub import HfApi

ROOT = pathlib.Path(__file__).resolve().parent.parent

SPACE_README = """---
title: Hinglish Intake
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: static
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 43200
hf_oauth_scopes:
  - read-repos
  - write-repos
hf_oauth_authorized_org: {org}
---

Upload calls, recordings and transcripts; a Mac worker processes them into
speaker-labelled transcripts, meeting notes and Claude-ready bundles.
Static Space: each teammate signs in with their own HF account and the page
reads/writes the private {org}/intake dataset directly.
"""


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    org = sys.argv[1]
    api = HfApi()
    me = api.whoami()["name"]
    data, space = f"{org}/intake", f"{org}/intake-app"
    print(f"as {me}: dataset {data}, space {space}")

    api.create_repo(data, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        shutil.copytree(ROOT / "team", t, dirs_exist_ok=True)
        claude_md = (t / "CLAUDE.md").read_text().replace("<org>/intake", data)
        (t / "CLAUDE.md").write_text(claude_md)
        (t / ".gitattributes").write_text("jobs/*/input/** filter=lfs diff=lfs merge=lfs -text\n"
                                          "*.jpg filter=lfs diff=lfs merge=lfs -text\n")
        api.upload_folder(folder_path=t, repo_id=data, repo_type="dataset",
                          commit_message="team workspace: CLAUDE.md + skills")
    print(f"  dataset ready: https://huggingface.co/datasets/{data}")

    # Static Space: free for organizations (Gradio/Docker Spaces in an org need a paid plan).
    api.create_repo(space, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        html = (ROOT / "app" / "index.html").read_text()
        org_id = next(o["id"] for o in api.whoami()["orgs"] if o["name"] == org)
        html = html.replace('const ORG = "your-org", ORG_ID = "";',
                            f'const ORG = "{org}", ORG_ID = "{org_id}";')
        (t / "index.html").write_text(html)
        (t / "README.md").write_text(SPACE_README.format(org=org))
        api.upload_folder(folder_path=t, repo_id=space, repo_type="space",
                          commit_message="Hinglish Intake app")
    print(f"  space ready:   https://huggingface.co/spaces/{space}")
    print(f"""
Left for you (once):
  1. Invite teammates to the {org} org with the "write" role (they upload jobs):
     https://huggingface.co/organizations/{org}/settings/members
  2. Start the worker on this Mac:  deploy/install_worker.sh {data}
""")


if __name__ == "__main__":
    main()
