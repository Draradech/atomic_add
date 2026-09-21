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
proton="/home/draradech/.local/share/Steam/steamapps/common/Proton 11.0/proton"
exe="$project_dir/target/release/wgsl-atomic-add-bench.exe"
prefix="/home/draradech/.cache/wgsl-atomic-proton"
csv="$project_dir/proton-$backend.csv"
csv_tmp="$csv.tmp"
log="$project_dir/proton-$backend.log"
native_fxc_source="/home/draradech/.local/share/Steam/steamapps/common/BeamNG.drive/Bin64/d3dcompiler_47.dll"
native_fxc="$project_dir/target/release/d3dcompiler_47.dll"

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

export STEAM_COMPAT_CLIENT_INSTALL_PATH="/home/draradech/.local/share/Steam"
export STEAM_COMPAT_DATA_PATH="$prefix"

if [[ "$backend" == "dx12" ]]; then
    # Proton's built-in D3DCompile rejects InterlockedAdd on the byte-address
    # buffer emitted by wgpu. Use Microsoft's FXC DLL, preserving the same
    # compiler path as the native Windows run.
    if [[ ! -f "$native_fxc_source" ]]; then
        echo "Microsoft FXC DLL not found: $native_fxc_source" >&2
        exit 1
    fi
    if [[ -e "$native_fxc" && ! "$native_fxc" -ef "$native_fxc_source" ]]; then
        echo "Refusing to replace a different DLL: $native_fxc" >&2
        exit 1
    fi
    if [[ ! -e "$native_fxc" ]]; then
        ln -s -- "$native_fxc_source" "$native_fxc"
    fi
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
    expected_header="buffer_bytes,buffer_mib,relation,power_of_two_exponent,average_gatomic_per_s,median_gatomic_per_s"
    IFS= read -r actual_header <"$csv_tmp" || actual_header=""
    actual_header=${actual_header%$'\r'}

    if [[ "$actual_header" != "$expected_header" ]]; then
        echo "Unexpected CSV header; leaving captured output at $csv_tmp" >&2
        exit 1
    fi

    mv -- "$csv_tmp" "$csv"
    echo "Completed: $csv" >&2
else
    status=$?
    echo "Benchmark failed with exit status $status; partial output is at $csv_tmp" >&2
    exit "$status"
fi
