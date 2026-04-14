import os

def fix_current_day_setter():
    # It might be in execution_loop.py or main.py depending on your architecture
    paths = ["core/execution_loop.py", "main.py"]
    for path in paths:
        if not os.path.exists(path):
            continue
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Change the read-only property assignment to the private backing variable
        target = "self.risk_manager.current_day ="
        if target in content:
            content = content.replace(target, "self.risk_manager._current_day =")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Fixed read-only property bug in {path}")

if __name__ == "__main__":
    fix_current_day_setter()
