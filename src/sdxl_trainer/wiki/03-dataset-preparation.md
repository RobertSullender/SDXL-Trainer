# Dataset preparation

## A working copy, never your originals

Click **New project** and enter a readable name, such as `Portrait study`. The app creates `~/SDXL-Trainer/projects/Portrait study` by default. The suggested name is `New project`, with a short number only if needed; project names never receive a generated hash. You can change the parent folder under **Models & runtime → Local runtime → Projects folder**.

Use **Open project** to choose an existing project. Import individual files or an entire folder using the app's matching dark file browser. Enable **Include subfolders** for recursive imports.

Every imported image is copied through a processing pipeline. EXIF orientation is applied. A valid embedded color profile is converted to sRGB before it is removed. Transparency is composited against the chosen fill color. A new RGB PNG is written from pixels alone: EXIF, GPS, text fields, and other source metadata are not carried over.

The original file remains untouched. A damaged image or invalid color profile causes an error rather than a silently changed color conversion. Keep your originals as your archival source.

Working files are named `Img-1.png`, `Img-2.png`, and so on. Captions use matching names such as `Img-1.txt`. Further imports continue the numbering.

To remove images from the draft, select one or more entries and choose **Remove selected…**, then confirm. Their prepared PNGs, draft captions, and project entries are removed together. The app renumbers the remaining pairs in their existing order, beginning at `Img-1`, and the next import takes the next consecutive number. In a consecutively numbered draft, removing `Img-515.png` makes the next pair `Img-515.png` and `Img-515.txt`. Gaps from earlier versions close the next time you remove a draft image. Originals and finalized exports keep their existing filenames. Finalize again to create a new snapshot reflecting the current draft.

## Resolution buckets

The closest supported aspect ratio is selected using the distance between aspect ratios on a logarithmic scale. This treats portrait and landscape symmetrically. The buckets have dimensions divisible by 64 and stay near SDXL's native one-megapixel area, from 512×2048 through 1024×1024 to 2048×512.

**Crop to fill** scales the image and center-crops excess pixels. It is the default. **Pad to fit** keeps the entire image and fills the remaining space with your chosen color. There is no AI-generated fill. Upscaling a small image cannot recover missing detail.

**Allow upscaling** is checked by default. Clear it before importing files or folders to skip images that need pixel enlargement for the selected fit. After orientation is applied, **Crop to fill** needs enlargement if either dimension is smaller than the bucket. **Pad to fit** needs enlargement if both dimensions are smaller. The import summary reports the skipped count.

The preview is the exact processed image used for captioning and training. No later random crop or flip changes its caption meaning. Fit/color changes apply to new imports. To use a different crop for an existing item, create a corrected source and import it into a new draft; this version does not include a manual crop editor.

## Finalizing

Review each caption, then click **Finalize dataset**. The app groups the prepared images by their existing resolution and writes each occupied bucket as a separate dataset in the next numbered revision under `finalized`. A project called `Photo-Realistic` with portrait and square buckets would export these folders on its first finalization.

```text
finalized/Dataset-1/Photo-Realistic:832X1216/
finalized/Dataset-1/Photo-Realistic:1024X1024/
```

If the combined name exceeds the filesystem's filename limit, the app shortens the project portion and keeps the full resolution suffix.

Each bucket folder contains its PNGs, full TXT captions, and a `manifest.json`. Finalization verifies the prepared image hashes, checks that captions are filled and reviewed, and records exact counts and fingerprints from both local tokenizers. The manifest seals the files with hashes and records dimensions, crop conditioning, token counts, and tokenizer identity. Original source paths stay in the working project.

The first bucket is selected automatically. Under **Training → Files → Choose project dataset…**, select a specific bucket or **Dataset-N / All aspect ratios**. The all-aspect-ratios option reads every bucket's existing images, captions, and manifest from that revision, including earlier exports. To select it with **Browse…**, choose the revision folder, such as `finalized/Dataset-1`. For a single bucket, browse to its folder containing `manifest.json`. Older mixed-resolution exports are also valid training datasets.

Training verifies the selected dataset again. Editing an exported image or caption invalidates its hash. Make edits in the working project and finalize again to create `finalized/Dataset-2`, `Dataset-3`, and so on. The bucket folders in each revision together capture the working dataset at the time of export. To resume a run, select the same revision and the same scope, either its individual bucket or all its aspect ratios.

## Fully purge a project

Open the project, then choose **Fully purge project…**. The app shows the exact project folder and asks you to type its name before permanently removing it. The project folder includes working images and captions, all its finalized dataset versions, project-contained training runs and checkpoints, and project logs. The displayed training log is cleared too. This deletion cannot be undone.

Files outside the project folder, including original images and shared model downloads in their usual locations, are preserved. Anything you put *inside* the project folder is deleted with it. Older training outputs, snapshots, or exports saved outside the project are not tracked or removed; manage those separately. Move or back up any project contents you want to keep before purging.

See [Captioning](04-captioning.md) and [Training](05-training.md).
