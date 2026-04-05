from app.services.catalog_service import CatalogService
from app.utils.config_utils import load_config


def main():
    config = load_config()
    catalog = CatalogService(config).load_or_build()
    cases = catalog.list_cases()
    print(f"Indexed {len(cases)} cases")


if __name__ == "__main__":
    main()
