def test_public_api_exports_resolve():
    import customer_seg
    for name in customer_seg.__all__:
        assert callable(getattr(customer_seg, name)) or isinstance(getattr(customer_seg, name), dict)
