
class ThompsonBudgetAllocator:
    def __init__(self, segment_df, platform_map, geo_map, tau=0.5, dist_type="normal"):
        self.df = segment_df.copy()
        self.platform_map = platform_map
        self.geo_map = geo_map
        self.df["platform"] = self.df["segment"].map(platform_map)
        self.df["geo"] = self.df["segment"].map(geo_map)
        self.total_budget = 1_000_000
        self.tau = tau
        self.dist_type = dist_type
        self.constraints = {
            "platform": {},
            "geo": {},
            "platform_geo": {}  # 这里新增联合约束
        }
        self._initialize_distributions()

    def _apply_group_constraints(self, df_alloc):
        # 平台单独约束
        for platform, limit in self.constraints.get("platform", {}).items():
            cond = df_alloc["platform"] == platform
            total = limit * self.total_budget
            df_alloc.loc[cond, "allocated_budget"] = (
                df_alloc.loc[cond, "allocated_budget"]
                .clip(upper=total)
            )

        # geo 单独约束
        for geo, limit in self.constraints.get("geo", {}).items():
            cond = df_alloc["geo"] == geo
            total = limit * self.total_budget
            df_alloc.loc[cond, "allocated_budget"] = (
                df_alloc.loc[cond, "allocated_budget"]
                .clip(upper=total)
            )

        # platform + geo 联合约束
        for (plat, geo), limit in self.constraints.get("platform_geo", {}).items():
            cond = (df_alloc["platform"] == plat) & (df_alloc["geo"] == geo)
            total = limit * self.total_budget
            df_alloc.loc[cond, "allocated_budget"] = (
                df_alloc.loc[cond, "allocated_budget"]
                .clip(upper=total)
            )

        return df_alloc
