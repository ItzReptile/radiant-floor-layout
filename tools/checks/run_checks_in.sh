#!/bin/bash
# Inside python:3.12-bookworm: /tb3 = TB3 checkout (ro), /task = the task (ro).
set -u
pip install -q PyYAML==6.0.2 >/dev/null 2>&1
curl -fsSL https://github.com/azu/dockerfile-pin/releases/download/v1.5.0/dockerfile-pin_linux_amd64.tar.gz -o /tmp/dp.tgz \
 && echo "88845bfbe4918596f00eb77ebeec8a0826026c9b24bcc83053cae48402e95ea6  /tmp/dp.tgz" | sha256sum --check --quiet \
 && tar -xzf /tmp/dp.tgz -C /usr/local/bin dockerfile-pin
W=/work; mkdir -p $W/tasks; cp -r /tb3/docs /tb3/scripts $W/ 2>/dev/null; cp -r /task $W/tasks/$NAME; cd $W
for c in scripts/checks/check-*.sh; do
  out=$(bash "$c" "tasks/$NAME" 2>&1); rc=$?
  printf '%-36s %s\n' "$(basename "$c")" "$([ $rc = 0 ] && echo PASS || echo "FAIL rc=$rc")"
  [ $rc = 0 ] || echo "$out" | grep -v '^Checking' | tail -14 | sed 's/^/    /'
done
