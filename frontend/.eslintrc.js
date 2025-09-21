module.exports = {
  extends: ['next/core-web-vitals'],
  rules: {
    // Disable problematic rules for faster development
    'react/no-unescaped-entities': 'off',
    'react-hooks/exhaustive-deps': 'warn', // Change from error to warning
    // Add any custom rules here
  },
}