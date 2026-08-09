# 怎么推上去（三条命令）

**⚠️ 我没有你的 GitHub 凭据，也不会要。下面三步你自己跑。**

## 1. 在 GitHub 网页上新建一个空仓库

```
名字      within-process-determinism
可见性    Public
⛔ 不要勾 "Add a README" / "Add .gitignore" / "Choose a license"
   —— 本地已经有了，勾了会冲突
```

## 2. 在本机这个文件夹里跑

```powershell
cd C:\Users\huxia\WorkBuddy\2026-06-19-23-05-14\dua-research\repo_within-process-determinism
git init
git add .
git commit -m "Measurement code and raw outputs for the within-process determinism note"
git branch -M main
git remote add origin https://github.com/<你的用户名>/within-process-determinism.git
git push -u origin main
```

## 3. 推完之后

```
□ 打开仓库首页，确认 README 正常显示
□ 点开 data/ 里五个文件，确认都能打开（不是 LFS 指针、不是乱码）
□ 把 URL 填回三处：
    ① METHODS_NOTE_FINAL 的 §7 与 §附A 的 ⟨REPOSITORY URL⟩
    ② README.md 里 citation 的 TODO
    ③ 个人页文案里的 ⟨GitHub URL⟩
```

## ⚠️ 推之前自己看一眼的两件事

```
□ RECORD.md 里的日期是我按文件时间和工作日志推的，你核对一遍
   —— 论文 §7 承诺了这份记录，日期错比没日期更糟
□ code/ 三个脚本里有没有本机路径、token、或任何你不想公开的东西
   （我扫过一遍没看到，但这是你自己该确认的）
```
