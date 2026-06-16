#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
JAR_ROOT="${HOME}/Library/DBeaverData/drivers/maven/maven-central"

ACCDB_PATH="${1:-${REPO_ROOT}/data/dapstom_6_4_combined_working_copy.accdb}"
OUT_DIR="${2:-${REPO_ROOT}/data/processed/dapstom_eda/tables}"

CP="${SCRIPT_DIR}"
CP="${CP}:${JAR_ROOT}/io.github.spannm/ucanaccess-5.1.5.jar"
CP="${CP}:${JAR_ROOT}/org.hsqldb/hsqldb-2.7.4.jar"
CP="${CP}:${JAR_ROOT}/io.github.spannm/jackcess-5.1.2.jar"
CP="${CP}:${JAR_ROOT}/org.apache.poi/poi-5.5.1.jar"
CP="${CP}:${JAR_ROOT}/commons-codec/commons-codec-1.20.0.jar"
CP="${CP}:${JAR_ROOT}/org.apache.commons/commons-collections4-4.5.0.jar"
CP="${CP}:${JAR_ROOT}/org.apache.commons/commons-math3-3.6.1.jar"
CP="${CP}:${JAR_ROOT}/commons-io/commons-io-2.21.0.jar"
CP="${CP}:${JAR_ROOT}/com.zaxxer/SparseBitSet-1.3.jar"
CP="${CP}:${JAR_ROOT}/org.apache.logging.log4j/log4j-api-2.24.3.jar"

mkdir -p "${OUT_DIR}"
javac -cp "${CP}" "${SCRIPT_DIR}/DapstomEdaExtractor.java"
java -cp "${CP}" DapstomEdaExtractor "${ACCDB_PATH}" "${OUT_DIR}"
