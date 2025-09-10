#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install_cmdstan.sh [options]

CmdStanの最新または指定バージョンをダウンロード・展開・（任意で）ビルドします。
既定では「cmdstan-<version>」というバージョン付きディレクトリに展開し、その中でビルドします。

Options:
  --version VERSION         特定のバージョンを指定 (例: 2.37.0)
  --install-dir DIR         インストール（配置）先ディレクトリ
                            （default: cmdstan-<version>）
  --no-build                ダウンロード・展開のみ実行（ビルドはスキップ）
  --log FILE                ログファイルパス (default: logs/cmdstan_install.log)
  -h, --help                このヘルプを表示

Examples:
  bash scripts/install_cmdstan.sh
  bash scripts/install_cmdstan.sh --version 2.37.0
  bash scripts/install_cmdstan.sh --version 2.37.0 --install-dir /opt/cmdstan-2.37.0
EOF
}

VERSION=""
INSTALL_DIR=""
NO_BUILD="false"
LOG=""
START_DIR="$(pwd)"

# ログファイルパスの決定とディレクトリ作成
if [[ -z "${LOG}" ]]; then
  LOG="logs/cmdstan_install.log"
fi
mkdir -p "$(dirname "$LOG")"

# ログ関数
log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG"
}

# 必要なコマンドの存在チェック
check_requirements() {
  local missing=()

  if ! command -v curl >/dev/null 2>&1; then
    missing+=("curl")
  fi

  # jq は VERSION 未指定（最新取得）時のみ必要
  if [[ -z "$VERSION" ]] && ! command -v jq >/dev/null 2>&1; then
    missing+=("jq")
  fi

  if ! command -v tar >/dev/null 2>&1; then
    missing+=("tar")
  fi

  if ! command -v make >/dev/null 2>&1; then
    missing+=("make")
  fi

  # 公式要件：モダンC++コンパイラ（g++ または clang++）
  if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
    missing+=("g++/clang++")
  fi

  if [[ ${#missing[@]} -gt 0 ]]; then
    log "ERROR: 以下のコマンドが見つかりません: ${missing[*]}"
    log "必要なパッケージをインストールしてください。"
    exit 1
  fi
}

# GitHub APIから最新バージョンを取得
get_latest_version() {
  local api_response
  if ! api_response=$(curl -k -fsSL  "https://api.github.com/repos/stan-dev/cmdstan/releases/latest"); then
    log "ERROR: GitHub APIへのアクセスに失敗しました" >&2
    exit 1
  fi

  local tag_name
  if ! tag_name=$(echo "$api_response" | jq -r '.tag_name'); then
    log "ERROR: バージョン情報の解析に失敗しました" >&2
    exit 1
  fi

  # "v" を外してバージョンのみ抽出（例: v2.37.0 → 2.37.0）
  echo "${tag_name#v}"
}

# バージョン情報の表示
show_version_info() {
  if [[ -n "$VERSION" ]]; then
    log "指定されたバージョン: ${VERSION}"
  else
    VERSION=$(get_latest_version)
    log "最新バージョン: ${VERSION}"
  fi
}

# ダウンロードURLの構築
build_download_url() {
  local version="$1"
  local base_url="https://github.com/stan-dev/cmdstan/releases/download"
  local filename="cmdstan-${version}.tar.gz"
  echo "${base_url}/v${version}/${filename}"
}

# ファイルのダウンロード
download_cmdstan() {
  local url="$1"
  local filename="$2"

  log "ダウンロード中: ${url}"
  if ! curl -k-fSLo "$filename" "$url"; then
    log "ERROR: ダウンロードに失敗しました"
    exit 1
  fi
  log "ダウンロード完了: ${filename}"
}

# アーカイブの展開（テンポラリに展開してから移動）
extract_archive() {
  local filename="$1"
  local extract_dir="$2"
  local tmpdir
  tmpdir="$(mktemp -d)"
  log "展開中: ${filename}"
  if ! tar -xzf "$filename" -C "$tmpdir"; then
    log "ERROR: アーカイブの展開に失敗しました"
    exit 1
  fi
  mv "$tmpdir/$extract_dir" .
  rmdir "$tmpdir"
  log "展開完了: ${extract_dir}"
}

# CmdStanのビルド
build_cmdstan() {
  local cmdstan_dir="$1"

  log "CmdStanをビルド中..."
  pushd "$cmdstan_dir" >/dev/null
  if ! make build; then
    log "ERROR: ビルドに失敗しました"
    popd >/dev/null
    exit 1
  fi

  # 成功確認：stanc のバージョンを記録（存在すれば）
  if [[ -x "bin/stanc" ]]; then
    ./bin/stanc --version 2>&1 | tee -a "$LOG" || true
  fi

  popd >/dev/null
  log "ビルド完了"
}

# インストール先の準備（既存があれば削除）
prepare_install_dir() {
  local install_dir="$1"
  if [[ -n "$install_dir" && -d "$install_dir" ]]; then
    log "既存のインストールを削除中: ${install_dir}"
    rm -rf "$install_dir"
  fi
}

# 絶対パス表示用に整形
abs_path_of_install_dir() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "${START_DIR}/$p"
  fi
}

# メイン処理
main() {
  log "== CmdStan インストール開始 =="

  # 事前チェック
  check_requirements

  # バージョン情報の取得・表示
  show_version_info

  # 既定の展開先/インストール先は「cmdstan-<version>」
  local extract_dir="cmdstan-${VERSION}"
  if [[ -z "$INSTALL_DIR" ]]; then
    INSTALL_DIR="$extract_dir"
  fi

  # 既存のINSTALL_DIRがある場合は削除
  prepare_install_dir "$INSTALL_DIR"

  # ダウンロードURLとファイル名
  local url
  url=$(build_download_url "$VERSION")
  local filename="cmdstan-${VERSION}.tar.gz"

  # ダウンロード
  download_cmdstan "$url" "$filename"

  # 展開
  extract_archive "$filename" "$extract_dir"

  # ビルド（オプション）
  if [[ "$NO_BUILD" != "true" ]]; then
    build_cmdstan "$extract_dir"
  else
    log "ビルドをスキップしました"
  fi

  # インストール先への移動・命名（必要な場合のみ）
  if [[ "$INSTALL_DIR" != "$extract_dir" ]]; then
    mkdir -p "$(dirname "$INSTALL_DIR")"
    log "インストール先に移動中: ${INSTALL_DIR}"
    mv "$extract_dir" "$INSTALL_DIR"
  fi

  # クリーンアップ
  log "一時ファイルを削除中..."
  rm -f "$filename"

  log "== インストール完了 =="
  local abs_install
  abs_install="$(abs_path_of_install_dir "$INSTALL_DIR")"
  log "インストール先: ${abs_install}"

  if [[ "$NO_BUILD" != "true" ]]; then
    log "CmdStanが正常にインストールされました。"
    log "使用例: $INSTALL_DIR/bin/stanc --help"
  else
    log "CmdStanがダウンロード・展開されました。"
    log "ビルドを実行するには: cd $INSTALL_DIR && make build"
  fi
}

# オプション解析
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="${2:-}"; shift 2;;
    --install-dir)
      INSTALL_DIR="${2:-}"; shift 2;;
    --no-build)
      NO_BUILD="true"; shift;;
    --log)
      LOG="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

# スクリプト実行
main "$@"