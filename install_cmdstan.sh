#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install_cmdstan.sh [options]

CmdStanの最新または指定バージョンをダウンロード・展開・（任意で）ビルドします。
既定では「$HOME/.cmdstan/cmdstan-<version>」に配置し、そのディレクトリ内でビルドします。

Options:
  --version VERSION         特定のバージョンを指定 (例: 2.37.0)
  --install-dir DIR         インストール（配置）先ディレクトリ
                            （default: $HOME/.cmdstan/cmdstan-<version>）
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
  if ! api_response=$(curl -fsSL --retry 3 --retry-delay 2 "https://api.github.com/repos/stan-dev/cmdstan/releases/latest"); then
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

# メイン処理
main() {
  log "== CmdStan インストール開始 =="

  # 事前チェック
  check_requirements

  # バージョン情報の取得・表示
  if [[ -n "$VERSION" ]]; then
    log "指定されたバージョン: ${VERSION}"
  else
    VERSION=$(get_latest_version)
    log "最新バージョン: ${VERSION}"
  fi

  # インストール先の決定
  local extract_dir="cmdstan-${VERSION}"
  if [[ -z "$INSTALL_DIR" ]]; then
    INSTALL_DIR="${HOME}/.cmdstan/${extract_dir}"
  fi

  # ダウンロードURLとファイル名
  local url="https://github.com/stan-dev/cmdstan/releases/download/v${VERSION}/cmdstan-${VERSION}.tar.gz"
  local filename="cmdstan-${VERSION}.tar.gz"
  local download_dir
  download_dir="$(mktemp -d)"

  # ダウンロード
  log "ダウンロード中: ${url}"
  if ! curl -k -fSL --max-time 300 --retry 3 --retry-delay 5 -o "${download_dir}/${filename}" "$url"; then
    log "ERROR: ダウンロードに失敗しました（3回リトライ後）"
    exit 1
  fi
  log "ダウンロード完了: ${download_dir}/${filename}"

  # 展開（直接インストール先に展開）
  log "展開中: ${download_dir}/${filename}"
  local tmpdir
  tmpdir="$(mktemp -d)"
  if ! tar -xzf "${download_dir}/${filename}" -C "$tmpdir"; then
    log "ERROR: アーカイブの展開に失敗しました"
    exit 1
  fi

  # インストール先の準備
  mkdir -p "$(dirname "$INSTALL_DIR")"
  if [[ -d "$INSTALL_DIR" ]]; then
    log "既存のインストール先を削除中: ${INSTALL_DIR}"
    rm -rf "$INSTALL_DIR"
  fi

  # インストール先に移動
  mv "$tmpdir/$extract_dir" "$INSTALL_DIR"
  rmdir "$tmpdir"
  log "展開完了: ${INSTALL_DIR}"

  # ビルド（オプション）
  if [[ "$NO_BUILD" != "true" ]]; then
    log "CmdStanをビルド中..."
    pushd "$INSTALL_DIR" >/dev/null
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
  else
    log "ビルドをスキップしました"
  fi

  # クリーンアップ
  log "一時ファイルを削除中..."
  rm -rf "$download_dir"

  log "== インストール完了 =="
  log "インストール先: ${INSTALL_DIR}"

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