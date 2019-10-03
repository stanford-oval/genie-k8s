
replace_config() {
sed \
  -e "s|@@JOB_NAME@@|${JOB_NAME}|g" \
  -e "s|@@OWNER@@|${OWNER}|g" \
  -e "s|@@IAM_ROLE@@|${IAM_ROLE}|g" \
  -e "s|@@IMAGE@@|${IMAGE}|g" \
  -e "s|@@experiment@@|${experiment}|g" \
  -e "s|@@dataset@@|${dataset}|g" \
  -e "s|@@model@@|${model}|g"
  "$@"
}
