## Установка окружения (Windows 10/11)

> Требуется **Miniconda / Anaconda 64-bit**  

### 1. Скачать Miniconda

```powershell
Invoke-WebRequest `
  -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" `
  -OutFile "$env:TEMP\Miniconda3-latest.exe"
```

### 2. Установить

```powershell
Start-Process "$env:TEMP\Miniconda3-latest.exe" -Wait
```
Установите в папку "C:\Miniconda3\"
При установке не ставьте галочку "Add to PATH".

### 3. Инициализировать PowerShell
```powershell
& "C:\Miniconda3\Scripts\conda.exe" init powershell
```
Перезагрузите powershell после инициализации conda в powershell

### 4. Создать активное окружение
```powershell
conda create -n migratetorch_rag python=3.12
conda activate migratetorch_rag
```

### 5. Установите зависимости в проекте
```powershell
conda env update -f environment.yml
```

### 6. Окружения для валидатора

Установка для окружения c pyTorch 1.0.0
```powershell
conda install pytorch-cpu==1.0.0 torchvision-cpu==0.2.1 cpuonly -c pytorch
```