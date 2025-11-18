import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import abspath, basename, dirname, exists

import wget

SCRIPT_DIR = dirname(abspath(__file__))

FILE_URLS = [
    (
        "B2K1Gamma_RapidSim_7TeV_K1KstarNonResonant_Tree.root",
        "https://cernbox.cern.ch/remote.php/dav/public-files/8mN10X8U7VGfaRc/B2K1Gamma_RapidSim_7TeV_K1KstarNonResonant_Tree.root",  # noqa: E501
    ),
    (
        "B2K1Gamma_RapidSim_7TeV_Tree.root",
        "https://cernbox.cern.ch/remote.php/dav/public-files/pr3aM8n2hPT4Pag/B2K1Gamma_RapidSim_7TeV_Tree.root",  # noqa: E501
    ),
    (
        "B2KstGamma_RapidSim_7TeV_KstarNonResonant_Tree.root",
        "https://cernbox.cern.ch/remote.php/dav/public-files/QuP2cHeISTTSLVv/B2KstGamma_RapidSim_7TeV_KstarNonResonant_Tree.root",  # noqa: E501
    ),
    (
        "B2KstGamma_RapidSim_7TeV_Tree.root",
        "https://cernbox.cern.ch/remote.php/dav/public-files/EH5yrCpGko7P7Mc/B2KstGamma_RapidSim_7TeV_Tree.root",  # noqa: E501
    ),
]


class DownloadProgressTracker:
    """Thread-safe progress tracker for multiple concurrent downloads."""

    def __init__(self, total_files):
        self.total_files = total_files
        self.completed_files = 0
        self.current_bytes = {}  # filename -> current bytes
        self.total_bytes = {}  # filename -> total bytes
        self.lock = threading.Lock()

    def update(self, filename, current, total):
        """Update progress for a specific file."""
        with self.lock:
            self.current_bytes[filename] = current
            self.total_bytes[filename] = total
            self._display_progress()

    def mark_complete(self, filename):
        """Mark a file as completed."""
        with self.lock:
            self.completed_files += 1
            # Remove from tracking
            self.current_bytes.pop(filename, None)
            self.total_bytes.pop(filename, None)
            self._display_progress()

    def _display_progress(self):
        """Display overall progress across all downloads."""
        # Calculate totals
        total_downloaded = sum(self.current_bytes.values())
        total_size = sum(self.total_bytes.values())

        # Files completed
        files_status = f"Files: {self.completed_files}/{self.total_files}"

        # Overall bytes
        if total_size > 0:
            total_downloaded_mb = int(total_downloaded / 1e6)
            total_size_mb = int(total_size / 1e6)
            percentage = (total_downloaded / total_size) * 100
            bytes_status = (
                f"{total_downloaded_mb:,}MB / {total_size_mb:,}MB ({percentage:.1f}%)"
            )
        else:
            bytes_status = "Initializing..."

        # Active downloads
        active_count = len(self.current_bytes)
        active_status = f"Active: {active_count}"

        # Combine and display
        progress_message = f"{files_status} | {bytes_status} | {active_status}"
        sys.stdout.write(f"\r{progress_message}")
        sys.stdout.flush()


def download_file(filename, url, progress_tracker):
    """Download a single file with progress tracking.

    Args:
        filename: Name of the file to download
        url: URL to download from
        progress_tracker: DownloadProgressTracker instance

    Returns:
        tuple: (filename, success: bool, message: str)
    """
    output_file = f"{SCRIPT_DIR}/{filename}"

    if exists(output_file):
        progress_tracker.mark_complete(filename)
        return filename, True, "Already exists"

    try:
        # Progress callback for this specific file
        def progress_callback(current, total, _width=80):
            progress_tracker.update(filename, current, total)

        wget.download(url=url, bar=progress_callback, out=output_file)
        progress_tracker.mark_complete(filename)
        return filename, True, "Downloaded successfully"
    except Exception as e:
        progress_tracker.mark_complete(filename)
        return filename, False, f"Failed: {e!s}"


def download_all_parallel(max_workers=None):
    """Download all files in parallel using ThreadPoolExecutor.

    Args:
        max_workers: Maximum number of concurrent downloads.
                    If None, defaults to number of files (but max 20).
    """
    if max_workers is None:
        max_workers = min(len(FILE_URLS), 20)

    print(
        f"Downloading {len(FILE_URLS)} files with {max_workers} parallel workers...\n"
    )

    # Create progress tracker
    progress_tracker = DownloadProgressTracker(len(FILE_URLS))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_file = {
            executor.submit(download_file, filename, url, progress_tracker): filename
            for filename, url in FILE_URLS
        }

        # Process completed downloads as they finish
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append((filename, False, f"Exception: {e!s}"))

    # Print summary
    print("\n\n" + "=" * 60)
    print("Download Summary:")
    print("=" * 60)

    successful = 0
    failed = 0
    for filename, file_success, message in results:
        status = "✓" if file_success else "✗"
        print(f"{status} {basename(filename)}: {message}")
        if file_success:
            successful += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Total: {successful} successful, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = download_all_parallel()
    sys.exit(0 if success else 1)
