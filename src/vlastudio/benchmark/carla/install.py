"""Install the pinned CARLA server explicitly, outside Python import/workflow calls."""

from pathlib import Path
from vlastudio.core.paths import cache_root as shared_cache

CARLA_URL = 'https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz'
MAPS_URL = 'https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz'
CARLA_SHA256 = '7b2f432ce74b251593f95dcc8dee6d99ad9625c2a71a2d6d48f24d879de19ef7'
MAPS_SHA256 = '49db78d14fef1f403f69fe1a13f1f1802f2111a6d98161826b2ae0d6dd96bba8'


B2D_REVISION = '7ec25d1c9f7522d923ce5f3420986cef1cb2d956'


def cache_root():
    return shared_cache() / "driving"


def source(download=True):
    from vlastudio.utils.assets import extract, fetch
    root = cache_root() / 'sources' / f'bench2drive-{B2D_REVISION[:12]}'
    if (root / '.vlastudio-asset.json').is_file():
        return root
    archive = fetch(f'https://github.com/Thinklab-SJTU/Bench2Drive/archive/{B2D_REVISION}.tar.gz',
                    cache_root() / 'downloads' / f'bench2drive-{B2D_REVISION}.tar.gz', download=download)
    return extract(archive, root, strip_root=True)


TCP_REVISION = '8a08b07883f10b7d83f6bf5dd475bda91a91c50a'


def tcp_source(download=True):
    from vlastudio.utils.assets import extract, fetch
    root = cache_root() / 'sources' / f'tcp-{TCP_REVISION[:12]}'
    if (root / '.vlastudio-asset.json').is_file():
        return root
    archive = fetch(f'https://github.com/Thinklab-SJTU/Bench2DriveZoo/archive/{TCP_REVISION}.tar.gz',
                    cache_root() / 'downloads' / f'tcp-{TCP_REVISION}.tar.gz', download=download)
    return extract(archive, root, strip_root=True)


def install(prefix=None, *, download=True, additional_maps=True):
    from vlastudio.utils.assets import extract, fetch, sha256
    import json
    root = Path(prefix).expanduser().resolve() if prefix else cache_root()/'runtimes/carla-0.9.15'
    if not root.joinpath('.vlastudio-asset.json').is_file():
        archive = fetch(CARLA_URL, cache_root()/'downloads/CARLA_0.9.15.tar.gz', checksum=CARLA_SHA256, download=download)
        extract(archive, root)
    if json.loads((root/'.vlastudio-asset.json').read_text())['archive_sha256'] != CARLA_SHA256:
        raise ValueError('CARLA runtime receipt differs from the pinned archive')
    root.chmod(0o755)
    if additional_maps and not root.joinpath('.vlastudio-maps.json').is_file():
        archive = fetch(MAPS_URL, cache_root()/'downloads/AdditionalMaps_0.9.15.tar.gz', checksum=MAPS_SHA256, download=download)
        unpacked = extract(archive, root.parent/'carla-0.9.15-additional-maps')
        import shutil
        # Upstream maps extend the package's CarlaUE4/Content hierarchy.
        def keep_newer(source, target):
            target = Path(target)
            if target.is_file() and target.stat().st_mtime >= Path(source).stat().st_mtime:
                return str(target)
            return shutil.copy2(source, target)
        shutil.copytree(unpacked, root, dirs_exist_ok=True, copy_function=keep_newer,
                        ignore=shutil.ignore_patterns('.vlastudio-asset.json'))
        (root/'.vlastudio-maps.json').write_text(json.dumps({'archive_sha256': sha256(archive)}, indent=2)+'\n')
    if additional_maps and json.loads((root/'.vlastudio-maps.json').read_text())['archive_sha256'] != MAPS_SHA256:
        raise ValueError('CARLA maps receipt differs from the pinned archive')
    if not root.joinpath('CarlaUE4.sh').is_file():
        raise FileNotFoundError('CARLA archive does not contain CarlaUE4.sh')
    root.chmod(0o755)
    return root


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--without-additional-maps', action='store_true')
    args = parser.parse_args()
    print(install(args.prefix, download=not args.offline,
                  additional_maps=not args.without_additional_maps))


if __name__ == '__main__':
    main()
