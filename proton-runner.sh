#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <dx12|vulkan>" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

backend=$1
case "$backend" in
    dx12 | vulkan) ;;
    *)
        usage
        exit 2
        ;;
esac

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
steam_dir=~/.local/share/Steam
proton="$steam_dir/steamapps/common/Proton 11.0/proton"
exe="$project_dir/wgsl_rust/target/release/wgsl-atomic-add-bench.exe"
prefix=~/.cache/wgsl-atomic-proton
csv="$project_dir/proton-$backend.csv"
csv_tmp="$csv.tmp"
log="$project_dir/proton-$backend.log"
native_fxc="$project_dir/wgsl_rust/target/release/d3dcompiler_47.dll"

is_microsoft_fxc() {
    local dll_info dll_strings resolved_dll install
    resolved_dll=$(readlink -f -- "$1") || return 1
    if [[ "$resolved_dll" == "$steam_dir/steamapps/common/"* ]]; then
        install=${resolved_dll#"$steam_dir/steamapps/common/"}
        case "${install%%/*}" in
            Proton* | GE-Proton*) return 1 ;;
        esac
    fi
    dll_info=$(file -Lb -- "$1") || return 1
    [[ "$dll_info" == *"PE32+"* && "$dll_info" == *"x86-64"* ]] || return 1
    # Wine's version resource also names Microsoft, so reject its compiler marker.
    dll_strings=$(strings -el -- "$1") || return 1
    [[ "$dll_strings" == *"Microsoft Corporation"* && "$dll_strings" != *"Wine D3DCompiler"* ]]
}

if [[ ! -x "$proton" ]]; then
    echo "Proton launcher not found or not executable: $proton" >&2
    exit 1
fi

if [[ ! -f "$exe" ]]; then
    echo "Windows benchmark executable not found: $exe" >&2
    exit 1
fi

if [[ -e "$csv" || -e "$csv_tmp" ]]; then
    echo "Refusing to overwrite an existing result or temporary file:" >&2
    echo "  $csv" >&2
    echo "  $csv_tmp" >&2
    echo "Move or remove it before running this benchmark again." >&2
    exit 1
fi

mkdir -p -- "$prefix"

export STEAM_COMPAT_CLIENT_INSTALL_PATH="$steam_dir"
export STEAM_COMPAT_DATA_PATH="$prefix"

if [[ "$backend" == "dx12" ]]; then
    # Proton's built-in D3DCompile rejects InterlockedAdd on the byte-address
    # buffer emitted by wgpu. Use Microsoft's FXC DLL, preserving the same
    # compiler path as the native Windows run.
    if ! command -v file >/dev/null || ! command -v strings >/dev/null; then
        echo "DX12 requires the file and strings commands to identify a Microsoft FXC DLL" >&2
        exit 1
    fi
    if [[ -e "$native_fxc" ]]; then
        if ! is_microsoft_fxc "$native_fxc"; then
            echo "Refusing to use a non-Microsoft or non-64-bit DLL: $native_fxc" >&2
            exit 1
        fi
    else
        native_fxc_source=""
        if [[ -d "$steam_dir/steamapps/common" ]]; then
            while IFS= read -r -d '' candidate; do
                if is_microsoft_fxc "$candidate"; then
                    native_fxc_source=$candidate
                    break
                fi
            done < <(find "$steam_dir/steamapps/common" -type f -iname 'd3dcompiler_47.dll' -print0)
        fi
        if [[ -z "$native_fxc_source" ]]; then
            echo "No 64-bit Microsoft d3dcompiler_47.dll found under $steam_dir/steamapps/common" >&2
            exit 1
        fi
        # Replace a broken link left by an older run after moving the Steam library.
        if [[ -L "$native_fxc" ]]; then
            rm -- "$native_fxc"
        fi
        ln -s -- "$native_fxc_source" "$native_fxc"
    fi
    echo "Microsoft FXC DLL: $(readlink -f -- "$native_fxc")" >&2
    export WGPU_DX12_COMPILER=fxc
    export WINEDLLOVERRIDES="d3dcompiler_47=n${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
fi

echo "Running Windows $backend backend through Proton 11.0" >&2
echo "Progress: $log" >&2
echo "Results:  $csv" >&2

# This is a console program, so launch Wine directly inside the prepared
# prefix. Proton's `run` verb goes through its Steam shim and detaches stdout.
if "$proton" runinprefix "$exe" --backend "$backend" \
    >"$csv_tmp" \
    2> >(tee "$log" >&2); then
    mv -- "$csv_tmp" "$csv"
    echo "Completed: $csv" >&2
else
    status=$?
    echo "Benchmark failed with exit status $status; partial output is at $csv_tmp" >&2
    exit "$status"
fi
