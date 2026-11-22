#!/usr/bin/env bash
# Wrap the PyInstaller folder dist/SigVerify into one AppImage (issue #42). From the
# repository root, after PyInstaller:
#     bash bundle/linux/build_appimage.sh v0.10.0
# appimagetool and the AppImage runtime are pinned and hash-checked, so one release always
# builds the same file. The current runtime mounts through FUSE (fusermount3 or fusermount).
set -euo pipefail

TAG="$1"
TOOL_URL=https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage
TOOL_SHA=ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0
RUNTIME_URL=https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64
RUNTIME_SHA=2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d
WORK=build/appimage

rm -rf "$WORK"
mkdir -p "$WORK/AppDir" dist
curl -fsSLo "$WORK/appimagetool" "$TOOL_URL"
curl -fsSLo "$WORK/runtime" "$RUNTIME_URL"
echo "$TOOL_SHA  $WORK/appimagetool" | sha256sum -c -
echo "$RUNTIME_SHA  $WORK/runtime" | sha256sum -c -
chmod +x "$WORK/appimagetool"

cp -r dist/SigVerify "$WORK/AppDir/SigVerify"
install -m 755 bundle/linux/AppRun "$WORK/AppDir/AppRun"
cp bundle/linux/sigverify.desktop "$WORK/AppDir/"
cp src/gui/assets/signature_icon.png "$WORK/AppDir/sigverify.png"

# Extract-and-run: the CI runner has no FUSE to mount appimagetool itself.
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 "$WORK/appimagetool" --runtime-file "$WORK/runtime" \
    "$WORK/AppDir" "dist/SigVerify-$TAG-x86_64.AppImage"
