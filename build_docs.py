def build():
    from subprocess import run
    from shutil import rmtree, copytree, make_archive
    from os.path import exists
    from pathlib import Path
    from re import sub

    # Build the book
    result = run("jupyter-book clean -a .", capture_output=True, shell=True)
    print(result.stdout.decode("utf-8"))
    print(result.stderr.decode("utf-8"))
    result = run("jupyter-book build --builder html .", capture_output=True, shell=True)
    print(result.stdout.decode("utf-8"))
    print(result.stderr.decode("utf-8"))


    # Copy to docs folder to publish to github pages
    if exists("docs"):
        rmtree("docs")
    copytree("_build/html", "docs")
    Path("docs/.nojekyll").touch()
    if exists("_static"):
        copytree("_static", "docs/_static", dirs_exist_ok=True)

    # Convert generated PNGs to WebP and rewrite HTML references
    from optimize_images import optimize
    optimize("docs")

if __name__ == '__main__':
    build()
