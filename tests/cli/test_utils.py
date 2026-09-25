from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field, model_validator

from microbleednet.cli.utils import (
    ConfigField,
    config_fields,
    load_config,
    parse_config,
)
from microbleednet.errors import ApplicationError, print_application_error
from microbleednet.orchestration.configs import IndexDataConfig


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ApplicationError, match="Configuration file not found"):
        load_config(tmp_path / "missing.toml")


def test_load_config_rejects_non_toml_suffix(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ApplicationError, match="must use the TOML format"):
        load_config(path)


def test_load_config_reports_missing_suffix(tmp_path: Path) -> None:
    path = tmp_path / "config"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ApplicationError, match="<none>"):
        load_config(path)


def test_load_config_rejects_malformed_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("not = valid = toml", encoding="utf-8")
    with pytest.raises(ApplicationError, match="Could not load configuration file"):
        load_config(path)


def test_parse_config_reports_validation_error_as_bad_parameter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text("dataset_dir = 'dataset'\n", encoding="utf-8")  # missing fields
    with pytest.raises(ApplicationError) as error:
        parse_config(path, IndexDataConfig, "index-data")

    assert "Configuration validation failed" in str(error.value)
    assert "microbleednet describe index-data" in str(error.value)


def test_parse_config_preserves_nested_application_error(tmp_path: Path) -> None:
    class InvalidConfig(BaseModel):
        @model_validator(mode="after")
        def reject_config(self) -> "InvalidConfig":
            raise ApplicationError(
                category="Manifest",
                summary="Upstream manifest is incomplete",
                fix="Rerun the producing stage",
            )

    path = tmp_path / "config.toml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ApplicationError) as error:
        parse_config(path, InvalidConfig, "example")

    assert error.value.category == "Manifest"
    assert error.value.summary == "Upstream manifest is incomplete"
    assert error.value.fix == "Rerun the producing stage"


def test_application_error_string_includes_optional_fields() -> None:
    error = ApplicationError(
        category="Configuration",
        summary="Configuration failed",
        cause="A field is invalid",
        fix="Correct the field",
    )

    assert str(error) == (
        "Configuration failed. Cause: A field is invalid. Fix: Correct the field."
    )


def test_application_error_string_supports_summary_only() -> None:
    assert str(ApplicationError(category="Internal", summary="Operation failed")) == (
        "Operation failed."
    )


def test_print_application_error_uses_plain_terminal_output(capsys) -> None:
    print_application_error(
        ApplicationError(
            category="Configuration",
            summary="Configuration failed",
            fix="Correct the field",
            context={"path": "config.toml"},
        )
    )

    assert capsys.readouterr().out == (
        "Configuration failed. Fix: Correct the field. "
        "Context: path: config.toml.\n"
    )


def test_print_application_error_without_context(capsys) -> None:
    print_application_error(
        ApplicationError(category="Internal", summary="Operation failed")
    )

    assert capsys.readouterr().out == "Operation failed.\n"


def test_config_fields_recurses_into_nested_models() -> None:
    class Inner(BaseModel):
        knob: int = Field(default=1, description="An inner knob.")

    class Outer(BaseModel):
        inner: Inner = Field(default_factory=Inner)

    fields = config_fields(Outer, prefix="")
    assert fields == [
        ConfigField(
            section="inner",
            key="knob",
            default="1",
            description="An inner knob.",
        )
    ]


def test_config_fields_reports_literal_options() -> None:
    class Config(BaseModel):
        modality: Literal["T2*-GRE", "SWI", "QSM"] = Field(
            default="T2*-GRE", description="Imaging modality."
        )

    fields = config_fields(Config, prefix="")

    assert fields == [
        ConfigField(
            section="",
            key="modality",
            default="'T2*-GRE'",
            description=(
                "Imaging modality. Allowed values: 'T2*-GRE', 'SWI', 'QSM'."
            ),
        )
    ]


def test_config_field_derives_required_state_from_default() -> None:
    required = ConfigField(section="", key="missing", default=None, description="")
    optional = ConfigField(section="", key="present", default="1", description="")

    assert required.default is None
    assert optional.default == "1"
