# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for TwirlingConfig.to_dict() and DebiasingConfig.to_dict() serialization."""

from pennylane_ionq.error_mitigation import (
    DebiasingConfig,
    OneQubitTwirling,
    PhiChiPattern,
    TwirlingConfig,
)


class TestTwirlingConfigToDict:
    """Tests for TwirlingConfig.to_dict()."""

    def test_defaults(self):
        """No pattern and NONE one-qubit twirling produces only the one_qubit_twirling key."""
        config = TwirlingConfig()
        assert config.to_dict() == {"one_qubit_twirling": "none"}

    def test_pattern_enum(self):
        """PhiChiPattern enum value is serialized to its string value."""
        config = TwirlingConfig(pattern=PhiChiPattern.STANDARD)
        result = config.to_dict()
        assert result["pattern"] == "standard"

    def test_pattern_string(self):
        """A plain string pattern is passed through unchanged."""
        config = TwirlingConfig(pattern="chi_only")
        result = config.to_dict()
        assert result["pattern"] == "chi_only"

    def test_pattern_none_omitted(self):
        """When pattern is None, the key is absent from the output."""
        config = TwirlingConfig(pattern=None)
        assert "pattern" not in config.to_dict()

    def test_one_qubit_enum(self):
        """OneQubitTwirling enum value is serialized to its string value."""
        config = TwirlingConfig(one_qubit=OneQubitTwirling.BOTH)
        assert config.to_dict()["one_qubit_twirling"] == "both"

    def test_one_qubit_string(self):
        """A plain string for one_qubit is passed through unchanged."""
        config = TwirlingConfig(one_qubit="decomposition")
        assert config.to_dict()["one_qubit_twirling"] == "decomposition"

    def test_all_fields(self):
        """All fields set produces a fully populated dict."""
        config = TwirlingConfig(pattern=PhiChiPattern.EXTENDED, one_qubit=OneQubitTwirling.ORDER)
        assert config.to_dict() == {"pattern": "extended", "one_qubit_twirling": "order"}


class TestDebiasingConfigToDict:
    """Tests for DebiasingConfig.to_dict()."""

    def test_defaults(self):
        """Default DebiasingConfig contains only debiasing=True."""
        config = DebiasingConfig()
        assert config.to_dict() == {"debiasing": True}

    def test_num_variants(self):
        """num_variants is included when set."""
        config = DebiasingConfig(num_variants=4)
        result = config.to_dict()
        assert result["debiasing"] is True
        assert result["num_variants"] == 4

    def test_num_variants_none_omitted(self):
        """num_variants key is absent when not set."""
        config = DebiasingConfig()
        assert "num_variants" not in config.to_dict()

    def test_twirling_included(self):
        """phi_chi_twirling key is present when twirling is set."""
        twirling = TwirlingConfig(pattern=PhiChiPattern.CHI_ONLY, one_qubit=OneQubitTwirling.NONE)
        config = DebiasingConfig(twirling=twirling)
        result = config.to_dict()
        assert result["phi_chi_twirling"] == {"pattern": "chi_only", "one_qubit_twirling": "none"}

    def test_twirling_none_omitted(self):
        """phi_chi_twirling key is absent when twirling is None."""
        config = DebiasingConfig()
        assert "phi_chi_twirling" not in config.to_dict()

    def test_all_fields(self):
        """All fields set produces a fully populated dict."""
        twirling = TwirlingConfig(
            pattern=PhiChiPattern.ALTERNATIVE, one_qubit=OneQubitTwirling.DECOMPOSITION
        )
        config = DebiasingConfig(num_variants=8, twirling=twirling)
        assert config.to_dict() == {
            "debiasing": True,
            "num_variants": 8,
            "phi_chi_twirling": {"pattern": "alternative", "one_qubit_twirling": "decomposition"},
        }
