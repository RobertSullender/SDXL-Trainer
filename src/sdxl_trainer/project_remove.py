"""Remove and renumber draft image/caption pairs."""
from dataclasses import replace
from pathlib import Path
import os
import shutil
import tempfile


def remove_images(project, image_ids):
    """Remove the selected pairs and number the survivors in draft order.

    Stage hard links only for paths that will change. If rollback fails, leave
    the staged files intact for manual recovery.
    """
    ids = list(image_ids)
    if not ids or any(type(value) is not int or value <= 0 for value in ids):
        raise ValueError("Select one or more valid images to remove.")
    project._validate_entries()
    selected = set(ids)
    old_entries = project.entries
    known_ids = {entry.id for entry in old_entries}
    for value in ids:
        if value not in known_ids:
            raise ValueError(f"Unknown image number: {value}")
    old_next_id = project.next_image_id
    image_dir = project.image_dir
    survivors = []
    affected = []
    renames = {}
    for entry in old_entries:
        image, caption = image_dir / entry.image, image_dir / entry.caption
        if entry.id in selected:
            affected.extend((image, caption))
            continue
        number = len(survivors) + 1
        if entry.id != number:
            entry = replace(entry, id=number, image=f"Img-{number}.png", caption=f"Img-{number}.txt")
            affected.extend((image, caption))
            renames[image] = image_dir / entry.image
            renames[caption] = image_dir / entry.caption
        survivors.append(entry)

    original_paths = set(affected)
    for target in renames.values():
        if target not in original_paths and (target.exists() or target.is_symlink()):
            raise ValueError(f"Cannot renumber the draft because {target.name} already exists.")

    stage = Path(tempfile.mkdtemp(prefix=".remove-", dir=project.root))
    vacated = []
    published = []
    keep_stage = False
    try:
        for path in affected:
            os.link(path, stage / path.name, follow_symlinks=False)
        for path in affected:
            path.unlink()
            vacated.append(path)
        for path, target in renames.items():
            # A new link refuses to overwrite any intervening file.
            os.link(stage / path.name, target, follow_symlinks=False)
            published.append(target)
        project.entries = survivors
        project.next_image_id = len(survivors) + 1
        project.save()
    except BaseException as exc:
        project.entries = old_entries
        project.next_image_id = old_next_id
        try:
            for target in reversed(published):
                target.unlink()
            for path in vacated:
                os.link(stage / path.name, path, follow_symlinks=False)
        except BaseException as rollback_exc:
            keep_stage = True
            raise RuntimeError(f"Image removal failed and recovery files remain in {stage}: {rollback_exc}") from exc
        raise
    finally:
        if not keep_stage:
            shutil.rmtree(stage, ignore_errors=True)
    return len(selected)
