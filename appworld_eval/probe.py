"""Phase-0 probe: confirms AppWorld is installed and the agent interaction loop works end-to-end.
Run with the 3.11 venv:  .venv311/bin/python -m appworld_eval.probe
This is the interface the PCCR+2D+DAG adapter (Phase 1) targets."""
from appworld import AppWorld, load_task_ids

def main():
    tids = load_task_ids("train")
    print(f"train tasks: {len(tids)}  first: {tids[0]}")
    with AppWorld(task_id=tids[0], experiment_name="pccr_dag_probe") as w:
        print("\nINSTRUCTION:", w.task.instruction)
        # the AGENT ACTION in AppWorld = execute Python code that calls apis.<app>.<method>(...)
        out = w.execute("print(apis.api_docs.show_app_descriptions())")
        print("\nEXECUTE output (first 300 chars):\n", str(out)[:300])
        print("\ntask_completed():", w.task_completed())
        # w.evaluate() -> metrics (call after the agent finishes)
    print("\nOK: AppWorld environment works end-to-end.")

if __name__ == "__main__":
    main()
