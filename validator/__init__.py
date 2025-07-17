import torch
import warnings
import traceback
import importlib


def test(name, fn):
    print(f"\nТест: {name}")
    try:
        fn()
        print(f"✅ OK: {name}\n")
    except Exception as e:
        print(f"❌ Ошибка: {name}")
        print(f"    {type(e).__name__}: {e}\n")

def run_tests():
    # Блокируем предупреждения, чтобы делать из них ошибки
    warnings.filterwarnings("error", category=UserWarning)
    warnings.filterwarnings("error", category=DeprecationWarning)
    
    tests = [
        ("torch._six", lambda: importlib.import_module("torch._six")),
        ("torch.qr()", lambda: torch.qr(torch.randn(3, 3))),
        ("torch.autograd.function.traceable", lambda: getattr(torch.autograd.function, "traceable")),
        ("torch.testing.make_non_contiguous", lambda: torch.testing.make_non_contiguous(torch.randn(2, 2))),
        ("gesv", lambda: torch.gesv(torch.randn(3, 1), torch.randn(3, 3))),
        ("potri", lambda: torch.potri(torch.eye(3), upper=True)),
        ("potrs", lambda: torch.potrs(torch.randn(3, 1), torch.randn(3, 3))),
        ("trtrs", lambda: torch.trtrs(torch.randn(3, 1), torch.randn(3, 3))),
    ]
    
    for name, fn in tests:
        test(name, fn)


if __name__ == "__main__":
    try:
        import torch
        print(f"PyTorch установлен: версия {torch.__version__}\n")
        run_tests()
    except():
        print('Нет факела')
