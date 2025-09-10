#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install_cmdstan.sh [options]

CmdStanの最新バージョンをダウンロード・ビルド・インストールします。

Options:
  --version VERSION         特定のバージョンを指定 (例: 2.37.0)
  --install-dir DIR         インストール先ディレクトリ (default: ./cmdstan)
  --no-build                ダウンロード・展開のみ実行（ビルドはスキップ）
  --log FILE                ログファイルパス (default: logs/cmdstan_install.log)
  -h, --help                このヘルプを表示

Examples:
  bash scripts/install_cmdstan.sh
  bash scripts/install_cmdstan.sh --version 2.37.0 --install-dir /opt/cmdstan
EOF
}

VERSION=""
INSTALL_DIR="./cmdstan"
NO_BUILD="false"
LOG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="${2:-}"; shift 2;;
    --install-dir)
      INSTALL_DIR="${2:-./cmdstan}"; shift 2;;
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

# 入力の妥当性チェック（必要に応じて追加）

# ログファイルパスの決定とディレクトリ作成
if [[ -z "$LOG" ]]; then
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
  
  if ! command -v jq >/dev/null 2>&1; then
    missing+=("jq")
  fi
  
  if ! command -v tar >/dev/null 2>&1; then
    missing+=("tar")
  fi
  
  if ! command -v make >/dev/null 2>&1; then
    missing+=("make")
  fi
  
  if [[ ${#missing[@]} -gt 0 ]]; then
    log "ERROR: 以下のコマンドが見つかりません: ${missing[*]}"
    log "必要なパッケージをインストールしてください。"
    exit 1
  fi
}

# GitHub APIから最新バージョンを取得
get_latest_version() {
  log "GitHub APIから最新バージョンを取得中..."
  local api_response
  if ! api_response=$(curl -s https://api.github.com/repos/stan-dev/cmdstan/releases/latest); then
    log "ERROR: GitHub APIへのアクセスに失敗しました"
    exit 1
  fi
  
  local tag_name
  if ! tag_name=$(echo "$api_response" | jq -r '.tag_name'); then
    log "ERROR: バージョン情報の解析に失敗しました"
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
  if ! curl -L -o "$filename" "$url"; then
    log "ERROR: ダウンロードに失敗しました"
    exit 1
  fi
  
  log "ダウンロード完了: ${filename}"
}

# アーカイブの展開
extract_archive() {
  local filename="$1"
  local extract_dir="$2"
  
  log "展開中: ${filename}"
  if ! tar -xzf "$filename"; then
    log "ERROR: アーカイブの展開に失敗しました"
    exit 1
  fi
  
  log "展開完了: ${extract_dir}"
}

# CmdStanのビルド
build_cmdstan() {
  local cmdstan_dir="$1"
  
  log "CmdStanをビルド中..."
  cd "$cmdstan_dir"
  
  if ! make build; then
    log "ERROR: ビルドに失敗しました"
    exit 1
  fi
  
  log "ビルド完了"
}

# インストール先の準備
prepare_install_dir() {
  local install_dir="$1"
  
  if [[ -d "$install_dir" ]]; then
    log "既存のインストールを削除中: ${install_dir}"
    rm -rf "$install_dir"
  fi
}

# メイン処理
main() {
  log "== CmdStan インストール開始 =="
  
  # 必要なコマンドのチェック
  check_requirements
  
  # バージョン情報の取得・表示
  show_version_info
  
  # インストール先の準備
  prepare_install_dir "$INSTALL_DIR"
  
  # ダウンロードURLの構築
  local url
  url=$(build_download_url "$VERSION")
  local filename="cmdstan-${VERSION}.tar.gz"
  local extract_dir="cmdstan-${VERSION}"
  
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
  
  # インストール先への移動
  if [[ "$INSTALL_DIR" != "./$extract_dir" ]]; then
    log "インストール先に移動中: ${INSTALL_DIR}"
    mv "$extract_dir" "$INSTALL_DIR"
  fi
  
  # クリーンアップ
  log "一時ファイルを削除中..."
  rm -f "$filename"
  
  log "== インストール完了 =="
  log "インストール先: $(pwd)/$INSTALL_DIR"
  
  if [[ "$NO_BUILD" != "true" ]]; then
    log "CmdStanが正常にインストールされました。"
    log "使用例: $INSTALL_DIR/bin/stanc --help"
  else
    log "CmdStanがダウンロード・展開されました。"
    log "ビルドを実行するには: cd $INSTALL_DIR && make build"
  fi
}

# スクリプト実行
main "$@"
