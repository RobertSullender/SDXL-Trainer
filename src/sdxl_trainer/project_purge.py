"""Preview and remove one confirmed project without following external links."""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid

from .paths import config_dir, data_dir


@dataclass(frozen=True)
class PurgePreview:
    path: Path
    files: int
    directories: int
    bytes: int
    device: int
    inode: int


def _mount_points() -> list[Path]:
    """Linux mountinfo also identifies bind mounts on the same filesystem."""
    try:
        lines = Path('/proc/self/mountinfo').read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        raise RuntimeError('Cannot inspect mounted filesystems; project purge is unavailable.') from exc
    result = []
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            raise RuntimeError('Cannot safely inspect mounted filesystems.')
        raw = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), parts[4])
        result.append(Path(raw))
    return result


def _reject_mounts(root: Path) -> None:
    for mount in _mount_points():
        if mount == root or root in mount.parents:
            raise ValueError(f'Project contains a mounted filesystem ({mount}); unmount it before purging.')


def _safe_project_path(root: str | Path) -> Path:
    value = Path(os.path.abspath(os.path.expanduser(os.fspath(root))))
    # Check every path component without following links, including the root.
    for path in (value, *value.parents):
        if path.is_symlink():
            raise ValueError('Project path cannot contain symbolic links.')
    home_checkout = (Path.home() / 'SDXL-Trainer').resolve()
    checkout = Path(__file__).resolve().parents[2]
    sensitive = [(data_dir() / 'models').resolve(), (data_dir() / 'runtime').resolve(), config_dir().resolve()]
    for repo in (home_checkout, checkout):
        sensitive.extend(repo / folder for folder in ('src', 'scripts', 'tests', 'packaging', 'docs', '.git'))
    protected = [Path('/'), Path.home().resolve(), home_checkout,
                 (home_checkout / 'projects').resolve(), checkout, *sensitive]
    if any(value == item or value in item.parents for item in protected):
        raise ValueError('Refusing to purge a home, workspace, repository, or system folder.')
    if any(item in value.parents for item in sensitive):
        raise ValueError('Refusing to purge application source, settings, runtime, or model files.')
    if (value / '.git').exists() or (value / '.git').is_symlink():
        raise ValueError('Refusing to purge a repository.')
    _reject_mounts(value)
    return value


def _open_project(root: Path):
    parent_fd = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        project_fd = os.open(root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        descriptor = os.open('project.json', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=project_fd)
        try:
            with os.fdopen(descriptor, 'rb') as stream:
                content = stream.read(16 * 1024 * 1024 + 1)
        except BaseException:
            raise
        if len(content) > 16 * 1024 * 1024:
            raise ValueError('Project manifest is too large to verify.')
        manifest = json.loads(content)
        if not isinstance(manifest, dict) or manifest.get('schema_version') != 1 or manifest.get('kind') != 'sdxl-trainer-draft':
            raise ValueError('Select a genuine SDXL-Trainer project folder.')
        images_fd = os.open('images', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=project_fd)
        os.close(images_fd)
        return parent_fd, project_fd
    except BaseException:
        os.close(project_fd)
        os.close(parent_fd)
        raise


def _count_tree(directory_fd: int, root_device: int) -> tuple[int, int, int]:
    files = directories = total = 0
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                if info.st_dev != root_device:
                    raise ValueError('Project contains another filesystem; unmount it before purging.')
                child_fd = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
                try:
                    child_files, child_dirs, child_bytes = _count_tree(child_fd, root_device)
                finally:
                    os.close(child_fd)
                files += child_files
                directories += 1 + child_dirs
                total += child_bytes
            else:
                files += 1
                total += info.st_size
    return files, directories, total


def preview_project_purge(root: str | Path) -> PurgePreview:
    """Inspect only the project tree. Symlinks are counted but never followed."""
    if not getattr(shutil.rmtree, 'avoids_symlink_attacks', False):
        raise RuntimeError('Safe project purge is unavailable on this platform.')
    path = _safe_project_path(root)
    parent_fd, project_fd = _open_project(path)
    try:
        info = os.fstat(project_fd)
        files, directories, total = _count_tree(project_fd, info.st_dev)
        return PurgePreview(path, files, directories, total, info.st_dev, info.st_ino)
    finally:
        os.close(project_fd)
        os.close(parent_fd)


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise OSError(errno.ENOSYS, 'Safe project purge requires Linux renameat2.')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), destination)


def purge_project(preview: PurgePreview) -> None:
    """Delete precisely the project that was previewed after explicit confirmation.

    Atomic renaming + identity checks prevent a substituted project from being
    removed. shutil's fd-based implementation does not follow symlinks. No
    source image, model, or separately located run is touched.
    """
    if not isinstance(preview, PurgePreview):
        raise TypeError('A project purge preview is required.')
    if not getattr(shutil.rmtree, 'avoids_symlink_attacks', False):
        raise RuntimeError('Safe project purge is unavailable on this platform.')
    path = _safe_project_path(preview.path)
    parent_fd, project_fd = _open_project(path)
    renamed = False
    quarantine = '.sdxl-purge-' + uuid.uuid4().hex
    try:
        info = os.fstat(project_fd)
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        expected = (preview.device, preview.inode)
        if (info.st_dev, info.st_ino) != expected or (current.st_dev, current.st_ino) != expected:
            raise ValueError('Project changed since the purge preview. Review it again.')
        _count_tree(project_fd, info.st_dev)
        _rename_noreplace(parent_fd, path.name, quarantine)
        renamed = True
        moved = os.stat(quarantine, dir_fd=parent_fd, follow_symlinks=False)
        if (moved.st_dev, moved.st_ino) != expected:
            raise ValueError('Project changed during purge. No files were deleted.')
        _reject_mounts(path.parent / quarantine)
        # rmtree opens relative to the trusted parent fd and never traverses links.
        shutil.rmtree(quarantine, dir_fd=parent_fd)
        renamed = False
    except BaseException as exc:
        if renamed:
            try:
                _rename_noreplace(parent_fd, quarantine, path.name)
            except OSError as restore_error:
                raise RuntimeError(f'Purge did not finish. Inspect {path.parent / quarantine}; it could not be restored: {restore_error}') from exc
        raise
    finally:
        os.close(project_fd)
        os.close(parent_fd)
