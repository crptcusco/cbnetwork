# Lista de tareas (Checklist) — Proyecto cbnetwork

Estado actual generado: 2026-01-13

- [x] Audit code for missing unit tests
- [x] Add unit tests for `directededge.py` (parser, evaluate, true table)
- [x] Add unit tests for `cnflist.py` (generate_cnf, simplify/remove duplicates)
- [x] Add unit tests for `localtemplates.py` (template generation functions)
- [x] Add unit tests for `globalnetwork` (state/attractor transforms)
- [x] Add unit tests for `globaltopology.py` (sample topology, plotting safety)
- [x] Add tests for CBN ordering/mounting methods (`order_edges_by_grade`, `mount_stable_attractor_fields`)
- [x] Add tests comparing brute-force vs SAT attractor finders for `LocalNetwork`
- [x] Run full test suite on clean environment / CI
- [ ] Run linters and static analysis (`flake8`, `mypy`)
- [ ] Apply code formatter (`black`) and ensure style consistency
- [ ] Pin and freeze dependencies (`requirements.txt` / `pyproject.toml`)
- [ ] Verify examples and notebooks run end-to-end
- [ ] Remove or gitignore large experiment files (or enable Git LFS)
- [ ] Update package version and prepare release tag
- [ ] Create release candidate and run smoke tests (install from sdist/wheel)
- [ ] Create distribution artifacts (`sdist`, `wheel`) and verify install
- [ ] Draft changelog entries (features, fixes, breaking changes)
- [ ] Finalize changelog and publish GitHub release

Notas:
- Tests locales actuales: `35 passed, 11 warnings`.
- Se añadió workflow CI básico en `.github/workflows/ci.yml` y script local `scripts/run_ci_locally.sh`.
- Recomendación: ejecutar linters y formateador antes del tag final; revisar notebooks en `examples/`.
