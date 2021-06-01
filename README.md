# TestGit
Just test git

# Personal Summary

## Normal flow

1. Initialize a local repository

```
git init
```

2. Add a remote

```
git remote add origin yourRemoteUrl
```

If there is already a remote:

```
git remote set-url origin git://new.url.here
```

or

```
git remote remove origin
git remote add origin yourRemoteUrl
```

Then

```
git push -u origin main
```

3. Git add changes

```
git add .
```

4. Commit with messages

```
git commit -m "My message"
```

5. Push to remote repository

For first time, we need to specify the origin brunch.

```
git push -u origin main
```

Otherwise, just

```
git push
```

## Other useful commands

update local repository

```
git pull
```

Check status (changes)

```
git status
```

Check modified logs

```
git log
```

