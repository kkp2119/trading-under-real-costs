"""run_holdout.is_clean (Kyle 2026-10-27, research 11 rule): `git -C <study> status --porcelain -- .` must print nothing.
A throwaway git repository under $R12_TEST_DIR stands in for the project (nothing in the real repository is touched)."""
import os
import subprocess

import run_holdout


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True,
                          text=True, check=True)


def test_clean_tree_rule(tmpdir_r12):
    repo = os.path.join(tmpdir_r12, "repo")
    study = os.path.join(repo, "research12")
    os.makedirs(os.path.join(study, "results"))
    g(tmpdir_r12, "init", "-q", repo)
    open(os.path.join(repo, ".gitignore"), "w").write("research12/results/HOLDOUT_UNLOCKED\nresearch12/bars1s/logs/\n")
    open(os.path.join(study, "a.py"), "w").write("x = 1\n")
    g(repo, "add", ".gitignore", "research12/a.py")
    g(repo, "commit", "-q", "-m", "init")
    assert run_holdout.is_clean(study)
    open(os.path.join(study, "results", "HOLDOUT_UNLOCKED"), "w").close()      # ignored: still clean
    os.makedirs(os.path.join(study, "bars1s", "logs"))
    open(os.path.join(study, "bars1s", "logs", "x.log"), "w").close()
    assert run_holdout.is_clean(study)
    open(os.path.join(repo, "outside.txt"), "w").close()                    # outside the study folder: not looked at
    assert run_holdout.is_clean(study)
    open(os.path.join(study, "new.tsv"), "w").close()                       # untracked inside: not clean
    assert not run_holdout.is_clean(study)
    os.rename(os.path.join(study, "new.tsv"), os.path.join(repo, "moved.tsv"))
    assert run_holdout.is_clean(study)
    open(os.path.join(study, "a.py"), "a").write("y = 2\n")                 # modified tracked: not clean
    assert not run_holdout.is_clean(study)
    assert not run_holdout.is_clean(os.path.join(tmpdir_r12, "not_a_repo_" + os.urandom(3).hex()))
