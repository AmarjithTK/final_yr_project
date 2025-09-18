Nice — if you want to run that Python script forever, the best way is to use **`tmux`** so it keeps running even after you close your terminal.

Here’s the **basic tmux workflow**:

---

### 🟢 Start a new session

```bash
tmux new -s mysession
```

* `-s mysession` gives it a name (you can call it whatever you want).
* Now you’re inside tmux.

---

### 🟢 Run your script

```bash
python3 yourscript.py
```

(or whichever script name you saved).

---

### 🟢 Detach (leave it running in background)

Press:

```
Ctrl + b   then   d
```

* This detaches you. Your script keeps running.

---

### 🟢 Reattach to the session

```bash
tmux attach -t mysession
```

---

### 🟢 List sessions (if you forget names)

```bash
tmux ls
```

---

### 🟢 Kill a session (stop it)

```bash
tmux kill-session -t mysession
```

---

👉 Do you want me to also show you a **one-liner command** that starts tmux, runs your Python script inside it, and detaches automatically?
