# Repository Instructions

## PR-Scoped Specifications

- Every pull request that adds a feature or changes or fixes existing behavior must include a specification.
- Create one specification per pull request at `specs/YYYY-MM-DD-<change-name>/spec.md`. Do not use a PR number in the directory name; record the PR number or URL inside the spec when available.
- Write the spec before implementation and include it in the same pull request as the code and tests. Keep it aligned with material requirement changes during review.
- Describe the goal and context, current and intended behavior, out-of-scope items, acceptance criteria, and verification approach. Document API, data format, and compatibility impacts when relevant; omit sections that do not apply.
- Link the spec from the pull request description. Keep a `specs/README.md` index linking to the specs.
- After a pull request is merged, keep its spec as historical record. Create a new spec for follow-up changes; do not rewrite a merged spec.
- Treat the existing root `spec.md` as the initial implementation's specification. Do not use it as the latest-spec file or update it for later pull requests.
