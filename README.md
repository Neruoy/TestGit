# TestGit
Just test git

# Personal Summary

## Normal flow

1. Initialize a local repository

```bash
git init
```

2. Add a remote

```bash
git remote add origin yourRemoteUrl
```

If there is already a remote:

```bash
git remote set-url origin git://new.url.here
```

or

```bash
git remote remove origin
git remote add origin yourRemoteUrl
```

Then

```bash
git push -u origin main
```

3. Git add changes

```bash
git add .
```

4. Commit with messages

```bash
git commit -m "My message"
```

5. Push to remote repository

For first time, we need to specify the origin brunch.

```bash
git push -u origin main
```

Otherwise, just

```bash
git push
```

## Other useful commands

update local repository

```bash
git pull
```

Check status (changes)

```bash
git status
```

Check modified logs

```bash
git log
```

