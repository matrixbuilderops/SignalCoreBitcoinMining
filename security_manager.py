"""
Security module for SignalCore Bitcoin Mining System.

This module provides security hardening, credential management, and
secure communication for production Bitcoin mining operations.
"""

import os
import json
import base64
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


logger = logging.getLogger(__name__)


@dataclass
class SecureCredentials:
    """Secure credential container."""

    rpc_user: str
    rpc_password: str
    wallet_name: str
    wallet_address: str
    encrypted: bool = False


class CredentialManager:
    """Secure credential management for Bitcoin mining operations."""

    def __init__(self, keyfile_path: str = ".mining_key"):
        """
        Initialize credential manager.

        Args:
            keyfile_path: Path to encryption key file
        """
        self.keyfile_path = keyfile_path
        self._encryption_key = None
        self._credentials = None

    def generate_encryption_key(self, password: str) -> bytes:
        """
        Generate encryption key from password.

        Args:
            password: User password

        Returns:
            Encryption key
        """
        password_bytes = password.encode()
        salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))

        # Store salt with key
        key_data = {"key": key.decode(), "salt": base64.b64encode(salt).decode()}

        with open(self.keyfile_path, "w") as f:
            json.dump(key_data, f)

        os.chmod(self.keyfile_path, 0o600)  # Read-only for owner
        return key

    def load_encryption_key(self, password: str) -> Optional[bytes]:
        """
        Load encryption key from file.

        Args:
            password: User password

        Returns:
            Encryption key or None if failed
        """
        try:
            if not os.path.exists(self.keyfile_path):
                return None

            with open(self.keyfile_path, "r") as f:
                key_data = json.load(f)

            salt = base64.b64decode(key_data["salt"])
            password_bytes = password.encode()

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )

            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            return key

        except Exception as e:
            logger.error(f"Failed to load encryption key: {e}")
            return None

    def encrypt_credentials(
        self, credentials: SecureCredentials, password: str
    ) -> bool:
        """
        Encrypt and store credentials.

        Args:
            credentials: Credentials to encrypt
            password: Encryption password

        Returns:
            True if successful
        """
        try:
            # Generate or load key
            key = self.load_encryption_key(password)
            if key is None:
                key = self.generate_encryption_key(password)

            fernet = Fernet(key)

            # Serialize credentials
            cred_data = {
                "rpc_user": credentials.rpc_user,
                "rpc_password": credentials.rpc_password,
                "wallet_name": credentials.wallet_name,
                "wallet_address": credentials.wallet_address,
            }

            # Encrypt
            encrypted_data = fernet.encrypt(json.dumps(cred_data).encode())

            # Store encrypted credentials
            encrypted_file = ".mining_credentials.enc"
            with open(encrypted_file, "wb") as f:
                f.write(encrypted_data)

            os.chmod(encrypted_file, 0o600)  # Read-only for owner

            logger.info("Credentials encrypted and stored securely")
            return True

        except Exception as e:
            logger.error(f"Failed to encrypt credentials: {e}")
            return False

    def decrypt_credentials(self, password: str) -> Optional[SecureCredentials]:
        """
        Decrypt and load credentials.

        Args:
            password: Decryption password

        Returns:
            Decrypted credentials or None
        """
        try:
            key = self.load_encryption_key(password)
            if key is None:
                return None

            fernet = Fernet(key)

            # Load encrypted credentials
            encrypted_file = ".mining_credentials.enc"
            if not os.path.exists(encrypted_file):
                return None

            with open(encrypted_file, "rb") as f:
                encrypted_data = f.read()

            # Decrypt
            decrypted_data = fernet.decrypt(encrypted_data)
            cred_data = json.loads(decrypted_data.decode())

            return SecureCredentials(
                rpc_user=cred_data["rpc_user"],
                rpc_password=cred_data["rpc_password"],
                wallet_name=cred_data["wallet_name"],
                wallet_address=cred_data["wallet_address"],
                encrypted=True,
            )

        except Exception as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            return None

    def load_credentials_from_file(
        self, filepath: str = "Bitcoin Core Node RPC.txt"
    ) -> Optional[SecureCredentials]:
        """
        Load credentials from plain text file (legacy support).

        Args:
            filepath: Path to credentials file

        Returns:
            Credentials or None
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Credentials file not found: {filepath}")
                return None

            with open(filepath, "r") as f:
                content = f.read()

            # Parse credentials
            lines = content.strip().split("\n")
            rpc_user = ""
            rpc_password = ""
            wallet_name = ""
            wallet_address = ""

            for line in lines:
                line = line.strip()
                if line.startswith("rpcuser="):
                    rpc_user = line.split("=", 1)[1]
                elif line.startswith("rpcpassword="):
                    rpc_password = line.split("=", 1)[1]
                elif "Wallet Name" in line or "wallet" in line.lower():
                    # Extract wallet name from context
                    if len(lines) > lines.index(line) + 1:
                        wallet_name = lines[lines.index(line) + 2].strip()
                elif line.startswith("bc1"):
                    wallet_address = line

            if not rpc_user or not rpc_password:
                logger.error("Invalid credentials format")
                return None

            return SecureCredentials(
                rpc_user=rpc_user,
                rpc_password=rpc_password,
                wallet_name=wallet_name or "SignalCoreBitcoinMining",
                wallet_address=wallet_address
                or "bc1qcmxyhlpm3lf9zvyevqas9n547lywws2e7wvuu1",
                encrypted=False,
            )

        except Exception as e:
            logger.error(f"Failed to load credentials from file: {e}")
            return None

    def get_secure_credentials(
        self, password: Optional[str] = None
    ) -> Optional[SecureCredentials]:
        """
        Get credentials using most secure available method.

        Args:
            password: Decryption password for encrypted credentials

        Returns:
            Secure credentials
        """
        # Try encrypted credentials first
        if password:
            credentials = self.decrypt_credentials(password)
            if credentials:
                return credentials

        # Fallback to plain text file
        credentials = self.load_credentials_from_file()
        if credentials:
            logger.warning("Using plain text credentials - consider encrypting them")
            return credentials

        logger.error("No credentials available")
        return None


class SecurityHardening:
    """Security hardening for Bitcoin mining system."""

    @staticmethod
    def secure_file_permissions() -> None:
        """Set secure file permissions for sensitive files."""
        sensitive_files = [
            "Bitcoin Core Node RPC.txt",
            ".mining_credentials.enc",
            ".mining_key",
            "mining_system.log",
        ]

        for filepath in sensitive_files:
            if os.path.exists(filepath):
                try:
                    os.chmod(filepath, 0o600)  # Read/write for owner only
                    logger.info(f"Secured file permissions: {filepath}")
                except Exception as e:
                    logger.warning(f"Could not secure {filepath}: {e}")

    @staticmethod
    def clear_environment_secrets() -> None:
        """Clear any secrets from environment variables."""
        sensitive_env_vars = [
            "RPC_PASSWORD",
            "BITCOIN_PASSWORD",
            "WALLET_PRIVATE_KEY",
            "MINING_PASSWORD",
        ]

        for var in sensitive_env_vars:
            if var in os.environ:
                del os.environ[var]
                logger.info(f"Cleared sensitive environment variable: {var}")

    @staticmethod
    def validate_system_security() -> Dict[str, bool]:
        """
        Validate system security configuration.

        Returns:
            Security validation results
        """
        validation = {
            "file_permissions_secure": True,
            "no_secrets_in_env": True,
            "secure_temp_dir": True,
            "log_file_secure": True,
        }

        # Check file permissions
        sensitive_files = ["Bitcoin Core Node RPC.txt", ".mining_credentials.enc"]
        for filepath in sensitive_files:
            if os.path.exists(filepath):
                stat = os.stat(filepath)
                if stat.st_mode & 0o077:  # Check if group/other have permissions
                    validation["file_permissions_secure"] = False
                    break

        # Check environment variables
        sensitive_patterns = ["password", "key", "secret", "token"]
        for var, value in os.environ.items():
            if any(pattern in var.lower() for pattern in sensitive_patterns):
                if any(pattern in value.lower() for pattern in sensitive_patterns):
                    validation["no_secrets_in_env"] = False
                    break

        # Check temp directory
        temp_dir = os.environ.get("TMPDIR", "/tmp")
        if os.path.exists(temp_dir):
            temp_stat = os.stat(temp_dir)
            if not (temp_stat.st_mode & 0o1000):  # Sticky bit
                validation["secure_temp_dir"] = False

        # Check log file
        if os.path.exists("mining_system.log"):
            log_stat = os.stat("mining_system.log")
            if log_stat.st_mode & 0o044:  # World readable
                validation["log_file_secure"] = False

        return validation

    @staticmethod
    def apply_security_hardening() -> bool:
        """
        Apply comprehensive security hardening.

        Returns:
            True if hardening successful
        """
        try:
            logger.info("Applying security hardening...")

            # Secure file permissions
            SecurityHardening.secure_file_permissions()

            # Clear environment secrets
            SecurityHardening.clear_environment_secrets()

            # Set secure umask
            os.umask(0o077)  # No permissions for group/other by default

            # Validate security
            validation = SecurityHardening.validate_system_security()

            if all(validation.values()):
                logger.info("✓ Security hardening applied successfully")
                return True
            else:
                logger.warning("⚠ Some security hardening failed")
                for check, result in validation.items():
                    if not result:
                        logger.warning(f"  - {check}: FAILED")
                return False

        except Exception as e:
            logger.error(f"Security hardening failed: {e}")
            return False


class SecureRPCClient:
    """Secure RPC client with credential protection."""

    def __init__(self, credential_manager: CredentialManager):
        """
        Initialize secure RPC client.

        Args:
            credential_manager: Credential manager instance
        """
        self.credential_manager = credential_manager
        self._credentials = None

    def authenticate(self, password: Optional[str] = None) -> bool:
        """
        Authenticate with Bitcoin Core RPC.

        Args:
            password: Optional password for encrypted credentials

        Returns:
            True if authentication successful
        """
        try:
            self._credentials = self.credential_manager.get_secure_credentials(password)
            return self._credentials is not None
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def get_rpc_auth(self) -> Optional[Tuple[str, str]]:
        """
        Get RPC authentication tuple.

        Returns:
            (username, password) tuple or None
        """
        if self._credentials:
            return (self._credentials.rpc_user, self._credentials.rpc_password)
        return None

    def get_wallet_info(self) -> Optional[Dict[str, str]]:
        """
        Get wallet information.

        Returns:
            Wallet info dictionary
        """
        if self._credentials:
            return {
                "wallet_name": self._credentials.wallet_name,
                "wallet_address": self._credentials.wallet_address,
            }
        return None

    def secure_logout(self) -> None:
        """Securely clear credentials from memory."""
        if self._credentials:
            # Overwrite sensitive data
            self._credentials.rpc_password = "x" * len(self._credentials.rpc_password)
            self._credentials = None


def setup_production_security(
    password: Optional[str] = None,
) -> Tuple[bool, CredentialManager]:
    """
    Setup production security environment.

    Args:
        password: Optional password for credential encryption

    Returns:
        (success, credential_manager) tuple
    """
    logger.info("Setting up production security environment...")

    try:
        # Apply security hardening
        hardening_success = SecurityHardening.apply_security_hardening()

        # Initialize credential manager
        credential_manager = CredentialManager()

        # Try to load credentials
        credentials = credential_manager.get_secure_credentials(password)
        if not credentials:
            logger.error("Failed to load credentials")
            return False, credential_manager

        # Encrypt credentials if not already encrypted and password provided
        if not credentials.encrypted and password:
            encryption_success = credential_manager.encrypt_credentials(
                credentials, password
            )
            if encryption_success:
                logger.info("✓ Credentials encrypted for enhanced security")
            else:
                logger.warning("⚠ Failed to encrypt credentials")

        # Validate final security state
        validation = SecurityHardening.validate_system_security()
        security_score = sum(validation.values()) / len(validation) * 100

        logger.info(f"Security setup complete - Score: {security_score:.0f}%")

        if security_score >= 75:  # Acceptable security level
            return True, credential_manager
        else:
            logger.warning("Security score below acceptable threshold")
            return False, credential_manager

    except Exception as e:
        logger.error(f"Security setup failed: {e}")
        return False, CredentialManager()


def get_security_status() -> Dict[str, Any]:
    """
    Get comprehensive security status.

    Returns:
        Security status dictionary
    """
    validation = SecurityHardening.validate_system_security()

    # Check for encrypted credentials
    encrypted_creds_exist = os.path.exists(".mining_credentials.enc")

    # Check for plain text credentials
    plaintext_creds_exist = os.path.exists("Bitcoin Core Node RPC.txt")

    security_score = sum(validation.values()) / len(validation) * 100

    return {
        "security_score": security_score,
        "validation_results": validation,
        "encrypted_credentials": encrypted_creds_exist,
        "plaintext_credentials": plaintext_creds_exist,
        "security_level": (
            "HIGH"
            if security_score >= 90
            else "MEDIUM" if security_score >= 75 else "LOW"
        ),
    }


if __name__ == "__main__":
    # Demo security setup
    print("SignalCore Bitcoin Mining - Security Demo")
    print("=" * 50)

    # Setup security
    success, cred_manager = setup_production_security()

    if success:
        print("✓ Security setup successful")

        # Create secure RPC client
        rpc_client = SecureRPCClient(cred_manager)
        if rpc_client.authenticate():
            print("✓ RPC authentication successful")

            wallet_info = rpc_client.get_wallet_info()
            if wallet_info:
                print(f"✓ Wallet: {wallet_info['wallet_name']}")
                print(f"✓ Address: {wallet_info['wallet_address'][:20]}...")

            rpc_client.secure_logout()
            print("✓ Secure logout completed")

    else:
        print("⚠ Security setup failed")

    # Show security status
    status = get_security_status()
    print(
        f"\nSecurity Status: {status['security_level']} ({status['security_score']:.0f}%)"
    )

    for check, result in status["validation_results"].items():
        status_symbol = "✓" if result else "✗"
        print(f"  {status_symbol} {check}")
