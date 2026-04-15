from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.registry import (
    _SCHEME_REGISTRY,
    register_scheme,
    get_scheme_class,
)
from vllm_ascend.quantization.methods.base import (
    AscendLinearScheme,
    AscendAttentionScheme,
    AscendMoEScheme,
)


class TestRegisterScheme(TestBase):

    def setUp(self):
        super().setUp()
        self._original_registry = _SCHEME_REGISTRY.copy()

    RegisterScheme1 = register_scheme("TEST_QUANT_TYPE", "linear")(MockLinearScheme)
    RegisterScheme2 = register_scheme("TEST_QUANT_TYPE", "moe")(MockMoEScheme)

    def test_register_scheme_registers_class(self):
        cls = get_scheme_class("TEST_QUANT_TYPE", "linear")
        self.assertIs(cls, MockLinearScheme)

    def test_register_scheme_registers_multiple_schemes(self):
        linear_cls = get_scheme_class("TEST_QUANT_TYPE", "linear")
        moe_cls = get_scheme_class("TEST_QUANT_TYPE", "moe")
        self.assertIs(linear_cls, MockLinearScheme)
        self.assertIs(moe_cls, MockMoEScheme)

    def test_register_duplicate_scheme_raises_value_error(self):
        with self.assertRaises(ValueError) as context:
            register_scheme("TEST_QUANT_TYPE", "linear")(MockLinearScheme)
        self.assertIn("already registered", str(context.exception))
        self.assertIn("TEST_QUANT_TYPE/linear", str(context.exception))

    def test_register_different_quant_types_same_layer_type(self):
        @register_scheme("TEST_QUANT_TYPE_2", "linear")
        class AnotherLinearScheme(AscendLinearScheme):
            def get_weight(self, input_size, output_size, params_dtype):
                return {}

            def apply(self, layer, x, bias=None, tp_rank=0):
                return x

        cls1 = get_scheme_class("TEST_QUANT_TYPE", "linear")
        cls2 = get_scheme_class("TEST_QUANT_TYPE_2", "linear")
        self.assertIs(cls1, MockLinearScheme)
        self.assertIs(cls2, AnotherLinearScheme)


class TestGetSchemeClass(TestBase):

    def test_get_scheme_class_existing_linear(self):
        cls = get_scheme_class("W8A8_DYNAMIC", "linear")
        self.assertIsNotNone(cls)
        self.assertTrue(issubclass(cls, AscendLinearScheme))

    def test_get_scheme_class_existing_moe(self):
        cls = get_scheme_class("W8A8_DYNAMIC", "moe")
        self.assertIsNotNone(cls)
        self.assertTrue(issubclass(cls, AscendMoEScheme))

    def test_get_scheme_class_existing_attention(self):
        cls = get_scheme_class("FAKQuant", "attention")
        self.assertIsNotNone(cls)

    def test_get_scheme_class_nonexistent(self):
        cls = get_scheme_class("NONEXISTENT", "linear")
        self.assertIsNone(cls)

    def test_get_scheme_class_nonexistent_layer_type(self):
        cls = get_scheme_class("W8A8_DYNAMIC", "nonexistent")
        self.assertIsNone(cls)

    def test_register_scheme_duplicate_raises(self):
        with self.assertRaises(ValueError):
            @register_scheme("W8A8_DYNAMIC", "linear")
            class Duplicate:
                pass

    def test_all_linear_schemes_subclass_ascend_linear_scheme(self):
        for (quant_type, layer_type), scheme_cls in _SCHEME_REGISTRY.items():
            if layer_type == "linear":
                self.assertTrue(
                    issubclass(scheme_cls, AscendLinearScheme),
                    f"{scheme_cls.__name__} for {quant_type}/{layer_type} "
                    f"should be subclass of AscendLinearScheme",
                )

    def test_all_moe_schemes_subclass_ascend_moe_scheme(self):
        for (quant_type, layer_type), scheme_cls in _SCHEME_REGISTRY.items():
            if layer_type == "moe":
                self.assertTrue(
                    issubclass(scheme_cls, AscendMoEScheme),
                    f"{scheme_cls.__name__} for {quant_type}/{layer_type} "
                    f"should be subclass of AscendMoEScheme",
                )

    def test_all_attention_schemes_subclass_ascend_attention_scheme(self):
        for (quant_type, layer_type), scheme_cls in _SCHEME_REGISTRY.items():
            if layer_type == "attention":
                self.assertTrue(
                    issubclass(scheme_cls, AscendAttentionScheme),
                    f"{scheme_cls.__name__} for {quant_type}/{layer_type} "
                    f"should be subclass of AscendAttentionScheme",
                )

    def test_registry_not_empty(self):
        self.assertGreater(len(_SCHEME_REGISTRY), 0)

    def test_registry_key_format(self):
        for key in _SCHEME_REGISTRY.keys():
            self.assertEqual(len(key), 2)
            self.assertIsInstance(key[0], str)
            self.assertIsInstance(key[1], str)
