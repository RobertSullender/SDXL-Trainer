# Dataset preparation

## A working copy, never your originals

Create a project in a new or empty folder, or open an existing project. Import individual files or an entire folder using the native file chooser. Enable **Include subfolders** for recursive imports.

Every imported image is copied through a processing pipeline. EXIF orientation is applied. A valid embedded color profile is converted to sRGB before it is removed. Transparency is composited against the chosen fill color. A new RGB PNG is written from pixels alone: EXIF, GPS, text fields, and other source metadata are not carried over.

The original file remains untouched. A damaged image or invalid color profile causes an error rather than a silently changed color conversion. Keep your originals as your archival source.

Working files are named `Img-1.png`, `Img-2.png`, and so on. Captions use matching names such as `Img-1.txt`. Further imports continue the numbering.

## Resolution buckets

The closest supported aspect ratio is selected using the distance between aspect ratios on a logarithmic scale. This treats portrait and landscape symmetrically. The buckets have dimensions divisible by 64 and stay near SDXL's native one-megapixel area, from 512×2048 through 1024×1024 to 2048×512.

**Crop to fill** scales the image and center-crops excess pixels. It is the default. **Pad to fit** keeps the entire image and fills the remaining space with your chosen color. There is no AI-generated fill. Upscaling a small image cannot recover missing detail.

The preview is the exact processed image used for captioning and training. No later random crop or flip changes its caption meaning. Fit/color changes apply to new imports. To use a different crop for an existing item, create a corrected source and import it into a new draft; this version does not include a manual crop editor.

## Finalizing

Review each caption, then click **Finalize dataset**, choose a parent folder, and enter a new folder name. The destination must not already exist. Finalization checks:

- Every image exists and still matches its prepared pixel hash.
- Each caption contains text and fits both actual SDXL tokenizer limits.
- Both local tokenizer assets are available; their fingerprints are recorded.
- Image and caption names match and all output files are safe, regular files.

The export contains PNGs, TXT captions, and one manifest. It omits your private original source paths. The manifest seals the content with hashes and records dimensions, crop conditioning, token counts, and tokenizer identity.

Training verifies this snapshot again. Editing an exported image or caption invalidates it: edit the draft and create another finalized snapshot instead. The manifest provides integrity checking, not operating-system write protection.

See [Captioning](04-captioning.md) and [Training](05-training.md).
