import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class DapstomEdaExtractor {
    private static final String DEFAULT_ACCDB =
        "data/dapstom_6_4_combined_working_copy.accdb";
    private static final String DEFAULT_OUT_DIR =
        "data/processed/dapstom_eda/tables";

    private static final List<String> EXPECTED_TABLES = List.of(
        "HAUL 6-4 COMBINED",
        "PREDATOR 6-4 COMBINED",
        "PREY 6-4 COMBINED",
        "PROVENANCE 6-4 COMBINED",
        "MURRAY_TAXONOMY",
        "QUALIFYER 6-4"
    );

    private static final class TableInfo {
        String name;
        String type;
        String remarks;
    }

    private static final class QuerySpec {
        String name;
        String sql;

        QuerySpec(String name, String sql) {
            this.name = name;
            this.sql = sql;
        }
    }

    private static final class CoreField {
        String table;
        String field;
        boolean text;

        CoreField(String table, String field, boolean text) {
            this.table = table;
            this.field = field;
            this.text = text;
        }
    }

    private static final class NumericFieldSpec {
        String variable;
        String table;
        String field;
        String unit;
        boolean logTransform;

        NumericFieldSpec(
            String variable,
            String table,
            String field,
            String unit,
            boolean logTransform
        ) {
            this.variable = variable;
            this.table = table;
            this.field = field;
            this.unit = unit;
            this.logTransform = logTransform;
        }
    }

    public static void main(String[] args) throws Exception {
        Path accdb = Path.of(args.length > 0 ? args[0] : DEFAULT_ACCDB).toAbsolutePath().normalize();
        Path outDir = Path.of(args.length > 1 ? args[1] : DEFAULT_OUT_DIR).toAbsolutePath().normalize();
        Files.createDirectories(outDir);

        String url = "jdbc:ucanaccess://" + accdb
            + ";memory=false;mirrorFolder=/private/tmp;skipIndexes=true;showSchema=true";

        Class.forName("net.ucanaccess.jdbc.UcanaccessDriver");
        System.out.println("Connecting to " + accdb);
        try (Connection conn = DriverManager.getConnection(url)) {
            conn.setReadOnly(true);
            System.out.println("Connected. Writing EDA tables to " + outDir);

            List<String[]> errors = new ArrayList<>();
            writeManifest(outDir, accdb, url);

            List<TableInfo> tables = loadTables(conn);
            writeTableInventory(outDir.resolve("table_inventory.csv"), tables);
            writeColumnInventory(conn, outDir.resolve("column_inventory.csv"), tables);
            writeRowCounts(conn, outDir.resolve("table_row_counts.csv"), tables, errors);
            writeMissingness(conn, outDir.resolve("critical_field_missingness.csv"), errors);
            writeNumericProfiles(
                conn,
                outDir.resolve("numeric_variable_summary.csv"),
                outDir.resolve("numeric_variable_histograms.csv"),
                errors
            );

            for (QuerySpec spec : queries()) {
                try {
                    writeQuery(conn, outDir.resolve(spec.name + ".csv"), spec.sql);
                    System.out.println("wrote " + spec.name + ".csv");
                } catch (SQLException e) {
                    errors.add(new String[] {spec.name, compact(spec.sql), e.getMessage()});
                    System.err.println("query failed: " + spec.name + " -> " + e.getMessage());
                }
            }

            writeErrors(outDir.resolve("query_errors.csv"), errors);
        }
    }

    private static List<QuerySpec> queries() {
        List<QuerySpec> q = new ArrayList<>();

        q.add(new QuerySpec("haul_temporal_by_year",
            "SELECT [Year] AS year, COUNT(*) AS haul_rows "
                + "FROM [HAUL 6-4 COMBINED] "
                + "GROUP BY [Year] ORDER BY [Year]"));

        q.add(new QuerySpec("haul_temporal_by_decade",
            "SELECT CAST(FLOOR([Year] / 10) * 10 AS INTEGER) AS decade, COUNT(*) AS haul_rows "
                + "FROM [HAUL 6-4 COMBINED] "
                + "WHERE [Year] IS NOT NULL "
                + "GROUP BY CAST(FLOOR([Year] / 10) * 10 AS INTEGER) "
                + "ORDER BY decade"));

        q.add(new QuerySpec("haul_spatial_by_sea",
            "SELECT [sea] AS sea, COUNT(*) AS haul_rows, "
                + "SUM(IIF([shot_lat_dd] IS NULL OR [shot_lon_dd] IS NULL, 0, 1)) AS with_lat_lon, "
                + "SUM(IIF([ices_rect] IS NULL OR [ices_rect]='', 0, 1)) AS with_ices_rect, "
                + "SUM(IIF([ices_division] IS NULL OR [ices_division]='', 0, 1)) AS with_ices_division, "
                + "COUNT(DISTINCT [ices_rect]) AS distinct_ices_rectangles "
                + "FROM [HAUL 6-4 COMBINED] "
                + "GROUP BY [sea] ORDER BY haul_rows DESC"));

        q.add(new QuerySpec("haul_spatial_resolution",
            "SELECT 'lat_lon' AS resolution, "
                + "SUM(IIF([shot_lat_dd] IS NULL OR [shot_lon_dd] IS NULL, 0, 1)) AS haul_rows, "
                + "COUNT(*) AS total_hauls FROM [HAUL 6-4 COMBINED] "
                + "UNION ALL SELECT 'ices_rectangle', "
                + "SUM(IIF([ices_rect] IS NULL OR [ices_rect]='', 0, 1)), COUNT(*) FROM [HAUL 6-4 COMBINED] "
                + "UNION ALL SELECT 'ices_division', "
                + "SUM(IIF([ices_division] IS NULL OR [ices_division]='', 0, 1)), COUNT(*) FROM [HAUL 6-4 COMBINED] "
                + "UNION ALL SELECT 'sea', "
                + "SUM(IIF([sea] IS NULL OR [sea]='', 0, 1)), COUNT(*) FROM [HAUL 6-4 COMBINED]"));

        q.add(new QuerySpec("provenance_source_summary",
            "SELECT prov.[source_type] AS source_type, "
                + "COUNT(DISTINCT prov.[cruise_name]) AS provenance_rows, "
                + "COUNT(DISTINCT h.[haul_id]) AS linked_hauls "
                + "FROM [PROVENANCE 6-4 COMBINED] prov "
                + "LEFT JOIN [HAUL 6-4 COMBINED] h ON prov.[cruise_name] = h.[cruise_name] "
                + "GROUP BY prov.[source_type] "
                + "ORDER BY provenance_rows DESC"));

        q.add(new QuerySpec("predator_pooling_summary",
            "SELECT [pooled] AS pooled, COUNT(*) AS predator_rows, "
                + "COUNT([num_stomachs]) AS with_num_stomachs, "
                + "SUM([num_stomachs]) AS total_stomachs, "
                + "AVG([num_stomachs]) AS avg_num_stomachs, "
                + "MIN([num_stomachs]) AS min_num_stomachs, "
                + "MAX([num_stomachs]) AS max_num_stomachs, "
                + "SUM([num_empty]) AS total_empty_stomachs "
                + "FROM [PREDATOR 6-4 COMBINED] "
                + "GROUP BY [pooled] ORDER BY predator_rows DESC"));

        q.add(new QuerySpec("predator_stomach_stats",
            "SELECT COUNT(*) AS predator_rows, COUNT([num_stomachs]) AS with_num_stomachs, "
                + "SUM([num_stomachs]) AS total_stomachs, AVG([num_stomachs]) AS avg_num_stomachs, "
                + "MIN([num_stomachs]) AS min_num_stomachs, MAX([num_stomachs]) AS max_num_stomachs, "
                + "COUNT([num_empty]) AS with_num_empty, SUM([num_empty]) AS total_empty_stomachs "
                + "FROM [PREDATOR 6-4 COMBINED]"));

        q.add(new QuerySpec("prey_evidence_stats",
            "SELECT COUNT(*) AS prey_rows, COUNT([min_num]) AS with_min_num, "
                + "SUM([min_num]) AS total_min_num, AVG([min_num]) AS avg_min_num, "
                + "MIN([min_num]) AS min_min_num, MAX([min_num]) AS max_min_num, "
                + "COUNT([cpw]) AS with_cpw, SUM([cpw]) AS total_cpw, AVG([cpw]) AS avg_cpw "
                + "FROM [PREY 6-4 COMBINED]"));

        q.add(new QuerySpec("prey_qualifiers",
            "SELECT prey.[qual_code] AS qual_code, q.[qual_description] AS qual_description, "
                + "COUNT(*) AS prey_rows, SUM(prey.[min_num]) AS total_min_num "
                + "FROM [PREY 6-4 COMBINED] prey "
                + "LEFT JOIN [QUALIFYER 6-4] q ON prey.[qual_code] = q.[qual_code] "
                + "GROUP BY prey.[qual_code], q.[qual_description] "
                + "ORDER BY prey_rows DESC"));

        q.add(new QuerySpec("taxonomy_coverage",
            "SELECT COUNT(*) AS prey_rows, "
                + "SUM(IIF(prey.[tsn] IS NULL, 0, 1)) AS with_prey_tsn, "
                + "SUM(IIF(mt.[tsn] IS NULL, 0, 1)) AS matched_murray_taxonomy, "
                + "SUM(IIF(mt.[aphiaid] IS NULL, 0, 1)) AS with_aphiaid, "
                + "COUNT(DISTINCT prey.[prey_name]) AS distinct_prey_names, "
                + "COUNT(DISTINCT prey.[tsn]) AS distinct_prey_tsn "
                + "FROM [PREY 6-4 COMBINED] prey "
                + "LEFT JOIN [MURRAY_TAXONOMY] mt ON prey.[tsn] = mt.[tsn]"));

        q.add(new QuerySpec("top_predators",
            "SELECT [pred] AS predator_name, [tsn] AS predator_tsn, COUNT(*) AS predator_rows, "
                + "SUM([num_stomachs]) AS total_stomachs, AVG([mean_length_cm]) AS avg_mean_length_cm "
                + "FROM [PREDATOR 6-4 COMBINED] "
                + "GROUP BY [pred], [tsn] "
                + "ORDER BY predator_rows DESC LIMIT 30"));

        q.add(new QuerySpec("top_prey",
            "SELECT [prey_name] AS prey_name, [tsn] AS prey_tsn, COUNT(*) AS prey_rows, "
                + "SUM([min_num]) AS total_min_num, SUM([cpw]) AS total_cpw "
                + "FROM [PREY 6-4 COMBINED] "
                + "GROUP BY [prey_name], [tsn] "
                + "ORDER BY prey_rows DESC LIMIT 30"));

        q.add(new QuerySpec("top_predator_prey_edges",
            "SELECT p.[pred] AS predator_name, p.[tsn] AS predator_tsn, "
                + "prey.[prey_name] AS prey_name, prey.[tsn] AS prey_tsn, "
                + "COUNT(*) AS prey_records, SUM(prey.[min_num]) AS total_min_num, "
                + "SUM(prey.[cpw]) AS total_cpw "
                + "FROM [PREDATOR 6-4 COMBINED] p "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "GROUP BY p.[pred], p.[tsn], prey.[prey_name], prey.[tsn] "
                + "ORDER BY prey_records DESC LIMIT 50"));

        q.add(new QuerySpec("top_predator_prey_edges_positive_prey_tsn",
            "SELECT p.[pred] AS predator_name, p.[tsn] AS predator_tsn, "
                + "prey.[prey_name] AS prey_name, prey.[tsn] AS prey_tsn, "
                + "COUNT(*) AS prey_records, SUM(prey.[min_num]) AS total_min_num, "
                + "SUM(prey.[cpw]) AS total_cpw "
                + "FROM [PREDATOR 6-4 COMBINED] p "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "WHERE prey.[tsn] > 0 "
                + "GROUP BY p.[pred], p.[tsn], prey.[prey_name], prey.[tsn] "
                + "ORDER BY prey_records DESC LIMIT 120"));

        q.add(new QuerySpec("global_network_summary",
            "SELECT COUNT(*) AS prey_records, "
                + "COUNT(DISTINCT p.[pred_id]) AS predator_records, "
                + "COUNT(DISTINCT p.[pred]) AS predator_names, "
                + "COUNT(DISTINCT p.[tsn]) AS predator_tsn, "
                + "COUNT(DISTINCT prey.[prey_name]) AS prey_names, "
                + "COUNT(DISTINCT prey.[tsn]) AS prey_tsn, "
                + "SUM(prey.[min_num]) AS total_min_num, "
                + "SUM(prey.[cpw]) AS total_cpw "
                + "FROM [PREDATOR 6-4 COMBINED] p "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id]"));

        q.add(new QuerySpec("global_network_summary_positive_prey_tsn",
            "SELECT COUNT(*) AS prey_records, "
                + "COUNT(DISTINCT p.[pred_id]) AS predator_records, "
                + "COUNT(DISTINCT p.[pred]) AS predator_names, "
                + "COUNT(DISTINCT p.[tsn]) AS predator_tsn, "
                + "COUNT(DISTINCT prey.[prey_name]) AS prey_names, "
                + "COUNT(DISTINCT prey.[tsn]) AS prey_tsn, "
                + "SUM(prey.[min_num]) AS total_min_num, "
                + "SUM(prey.[cpw]) AS total_cpw "
                + "FROM [PREDATOR 6-4 COMBINED] p "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "WHERE prey.[tsn] > 0"));

        q.add(new QuerySpec("global_unique_edge_pairs",
            "SELECT COUNT(*) AS unique_predator_prey_pairs "
                + "FROM ("
                + "SELECT p.[pred] AS predator_name, prey.[prey_name] AS prey_name "
                + "FROM [PREDATOR 6-4 COMBINED] p "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "GROUP BY p.[pred], prey.[prey_name]"
                + ") x"));

        q.add(new QuerySpec("global_unique_edge_pairs_positive_prey_tsn",
            "SELECT COUNT(*) AS unique_predator_prey_pairs_positive_prey_tsn "
                + "FROM ("
                + "SELECT p.[pred] AS predator_name, prey.[prey_name] AS prey_name "
                + "FROM [PREDATOR 6-4 COMBINED] p "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "WHERE prey.[tsn] > 0 "
                + "GROUP BY p.[pred], prey.[prey_name]"
                + ") x"));

        q.add(new QuerySpec("negative_tsn_prey_categories",
            "SELECT [tsn] AS prey_tsn, [prey_name] AS prey_name, "
                + "COUNT(*) AS prey_rows, SUM([min_num]) AS total_min_num, SUM([cpw]) AS total_cpw "
                + "FROM [PREY 6-4 COMBINED] "
                + "WHERE [tsn] < 0 "
                + "GROUP BY [tsn], [prey_name] "
                + "ORDER BY prey_rows DESC"));

        q.add(new QuerySpec("network_potential_by_sea_decade",
            "SELECT h.[sea] AS sea, CAST(FLOOR(h.[Year] / 10) * 10 AS INTEGER) AS decade, "
                + "COUNT(DISTINCT h.[haul_id]) AS hauls, "
                + "COUNT(DISTINCT p.[pred_id]) AS predator_records, "
                + "COUNT(*) AS prey_records, "
                + "COUNT(DISTINCT p.[pred]) AS predator_names, "
                + "COUNT(DISTINCT prey.[prey_name]) AS prey_names "
                + "FROM ([HAUL 6-4 COMBINED] h "
                + "JOIN [PREDATOR 6-4 COMBINED] p ON h.[haul_id] = p.[haul_id]) "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "WHERE h.[Year] IS NOT NULL "
                + "GROUP BY h.[sea], CAST(FLOOR(h.[Year] / 10) * 10 AS INTEGER) "
                + "ORDER BY decade, prey_records DESC"));

        q.add(new QuerySpec("edge_pairs_by_sea_decade",
            "SELECT sea, decade, COUNT(*) AS unique_predator_prey_pairs, "
                + "SUM(prey_records) AS prey_records "
                + "FROM ("
                + "SELECT h.[sea] AS sea, CAST(FLOOR(h.[Year] / 10) * 10 AS INTEGER) AS decade, "
                + "p.[pred] AS predator_name, prey.[prey_name] AS prey_name, COUNT(*) AS prey_records "
                + "FROM ([HAUL 6-4 COMBINED] h "
                + "JOIN [PREDATOR 6-4 COMBINED] p ON h.[haul_id] = p.[haul_id]) "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "WHERE h.[Year] IS NOT NULL "
                + "GROUP BY h.[sea], CAST(FLOOR(h.[Year] / 10) * 10 AS INTEGER), p.[pred], prey.[prey_name]"
                + ") x GROUP BY sea, decade "
                + "ORDER BY unique_predator_prey_pairs DESC LIMIT 100"));

        q.add(new QuerySpec("network_potential_by_sea_decade_positive_prey_tsn",
            "SELECT h.[sea] AS sea, CAST(FLOOR(h.[Year] / 10) * 10 AS INTEGER) AS decade, "
                + "COUNT(DISTINCT h.[haul_id]) AS hauls, "
                + "COUNT(DISTINCT p.[pred_id]) AS predator_records, "
                + "COUNT(*) AS prey_records, "
                + "COUNT(DISTINCT p.[pred]) AS predator_names, "
                + "COUNT(DISTINCT prey.[prey_name]) AS prey_names "
                + "FROM ([HAUL 6-4 COMBINED] h "
                + "JOIN [PREDATOR 6-4 COMBINED] p ON h.[haul_id] = p.[haul_id]) "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "WHERE h.[Year] IS NOT NULL AND prey.[tsn] > 0 "
                + "GROUP BY h.[sea], CAST(FLOOR(h.[Year] / 10) * 10 AS INTEGER) "
                + "ORDER BY decade, prey_records DESC"));

        q.add(new QuerySpec("edge_pairs_by_sea_decade_positive_prey_tsn",
            "SELECT sea, decade, COUNT(*) AS unique_predator_prey_pairs, "
                + "SUM(prey_records) AS prey_records "
                + "FROM ("
                + "SELECT h.[sea] AS sea, CAST(FLOOR(h.[Year] / 10) * 10 AS INTEGER) AS decade, "
                + "p.[pred] AS predator_name, prey.[prey_name] AS prey_name, COUNT(*) AS prey_records "
                + "FROM ([HAUL 6-4 COMBINED] h "
                + "JOIN [PREDATOR 6-4 COMBINED] p ON h.[haul_id] = p.[haul_id]) "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "WHERE h.[Year] IS NOT NULL AND prey.[tsn] > 0 "
                + "GROUP BY h.[sea], CAST(FLOOR(h.[Year] / 10) * 10 AS INTEGER), p.[pred], prey.[prey_name]"
                + ") x GROUP BY sea, decade "
                + "ORDER BY unique_predator_prey_pairs DESC"));

        q.add(new QuerySpec("sea_decade_network_edges_positive_prey_tsn",
            "SELECT h.[sea] AS sea, CAST(FLOOR(h.[Year] / 10) * 10 AS INTEGER) AS decade, "
                + "p.[pred] AS predator_name, p.[tsn] AS predator_tsn, "
                + "prey.[prey_name] AS prey_name, prey.[tsn] AS prey_tsn, "
                + "COUNT(*) AS prey_records, COUNT(DISTINCT h.[haul_id]) AS distinct_hauls, "
                + "SUM(prey.[min_num]) AS total_min_num, "
                + "SUM(prey.[cpw]) AS total_cpw "
                + "FROM ([HAUL 6-4 COMBINED] h "
                + "JOIN [PREDATOR 6-4 COMBINED] p ON h.[haul_id] = p.[haul_id]) "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "WHERE h.[Year] IS NOT NULL AND prey.[tsn] > 0 "
                + "GROUP BY h.[sea], CAST(FLOOR(h.[Year] / 10) * 10 AS INTEGER), "
                + "p.[pred], p.[tsn], prey.[prey_name], prey.[tsn] "
                + "ORDER BY h.[sea], decade, prey_records DESC"));

        q.add(new QuerySpec("data_quality_checks",
            "SELECT 'duplicate_haul_id_rows' AS check_name, "
                + "COUNT(*) - COUNT(DISTINCT [haul_id]) AS flagged_rows "
                + "FROM [HAUL 6-4 COMBINED] "
                + "UNION ALL SELECT 'duplicate_pred_id_rows', "
                + "COUNT(*) - COUNT(DISTINCT [pred_id]) "
                + "FROM [PREDATOR 6-4 COMBINED] "
                + "UNION ALL SELECT 'duplicate_prey_id_rows', "
                + "COUNT(*) - COUNT(DISTINCT [id]) "
                + "FROM [PREY 6-4 COMBINED] "
                + "UNION ALL SELECT 'predator_orphan_haul_id', COUNT(*) "
                + "FROM [PREDATOR 6-4 COMBINED] p "
                + "LEFT JOIN [HAUL 6-4 COMBINED] h ON p.[haul_id] = h.[haul_id] "
                + "WHERE h.[haul_id] IS NULL "
                + "UNION ALL SELECT 'prey_orphan_pred_id', COUNT(*) "
                + "FROM [PREY 6-4 COMBINED] prey "
                + "LEFT JOIN [PREDATOR 6-4 COMBINED] p ON prey.[pred_id] = p.[pred_id] "
                + "WHERE p.[pred_id] IS NULL "
                + "UNION ALL SELECT 'haul_without_ices_row', COUNT(*) "
                + "FROM [HAUL 6-4 COMBINED] h "
                + "LEFT JOIN [ICES 6-4 COMBINED] i ON h.[haul_id] = i.[hauls_id] "
                + "WHERE i.[hauls_id] IS NULL "
                + "UNION ALL SELECT 'num_empty_gt_num_stomachs', COUNT(*) "
                + "FROM [PREDATOR 6-4 COMBINED] "
                + "WHERE [num_empty] IS NOT NULL AND [num_stomachs] IS NOT NULL "
                + "AND [num_empty] > [num_stomachs] "
                + "UNION ALL SELECT 'negative_min_num', COUNT(*) "
                + "FROM [PREY 6-4 COMBINED] WHERE [min_num] < 0 "
                + "UNION ALL SELECT 'negative_cpw', COUNT(*) "
                + "FROM [PREY 6-4 COMBINED] WHERE [cpw] < 0 "
                + "UNION ALL SELECT 'fractional_min_num', COUNT(*) "
                + "FROM [PREY 6-4 COMBINED] "
                + "WHERE [min_num] IS NOT NULL AND [min_num] <> FLOOR([min_num]) "
                + "UNION ALL SELECT 'fractional_num_empty', COUNT(*) "
                + "FROM [PREDATOR 6-4 COMBINED] "
                + "WHERE [num_empty] IS NOT NULL AND [num_empty] <> FLOOR([num_empty]) "
                + "UNION ALL SELECT 'latitude_out_of_range', COUNT(*) "
                + "FROM [HAUL 6-4 COMBINED] "
                + "WHERE [shot_lat_dd] IS NOT NULL AND ([shot_lat_dd] < -90 OR [shot_lat_dd] > 90) "
                + "UNION ALL SELECT 'longitude_out_of_range', COUNT(*) "
                + "FROM [HAUL 6-4 COMBINED] "
                + "WHERE [shot_lon_dd] IS NOT NULL AND ([shot_lon_dd] < -180 OR [shot_lon_dd] > 180)"));

        q.add(new QuerySpec("joined_record_sample",
            "SELECT h.[haul_id] AS haul_id, h.[cruise_name] AS cruise_name, h.[Year] AS year, "
                + "h.[sea] AS sea, h.[ices_rect] AS ices_rect, h.[shot_lat_dd] AS shot_lat_dd, "
                + "h.[shot_lon_dd] AS shot_lon_dd, p.[pred_id] AS pred_id, p.[pred] AS predator_name, "
                + "p.[tsn] AS predator_tsn, p.[pooled] AS pooled, p.[num_stomachs] AS num_stomachs, "
                + "prey.[prey_name] AS prey_name, prey.[tsn] AS prey_tsn, prey.[qual_code] AS qual_code, "
                + "prey.[min_num] AS min_num, prey.[cpw] AS cpw "
                + "FROM ([HAUL 6-4 COMBINED] h "
                + "JOIN [PREDATOR 6-4 COMBINED] p ON h.[haul_id] = p.[haul_id]) "
                + "JOIN [PREY 6-4 COMBINED] prey ON p.[pred_id] = prey.[pred_id] "
                + "LIMIT 100"));

        return q;
    }

    private static List<TableInfo> loadTables(Connection conn) throws SQLException {
        DatabaseMetaData md = conn.getMetaData();
        List<TableInfo> tables = new ArrayList<>();
        try (ResultSet rs = md.getTables(null, null, "%", new String[] {"TABLE", "VIEW"})) {
            while (rs.next()) {
                String name = rs.getString("TABLE_NAME");
                if (name == null || name.startsWith("MSys")) {
                    continue;
                }
                TableInfo t = new TableInfo();
                t.name = name;
                t.type = rs.getString("TABLE_TYPE");
                t.remarks = rs.getString("REMARKS");
                tables.add(t);
            }
        }
        tables.sort(Comparator.comparing((TableInfo t) -> t.type).thenComparing(t -> t.name));
        return tables;
    }

    private static void writeManifest(Path outDir, Path accdb, String url) throws IOException {
        Path p = outDir.resolve("eda_manifest.json");
        String sanitizedUrl = url.replace(accdb.toString(), "<ACCDB_PATH>");
        try (BufferedWriter w = Files.newBufferedWriter(p, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"generated_at_utc\": \"" + escapeJson(Instant.now().toString()) + "\",\n");
            w.write("  \"source_accdb\": \"" + escapeJson(accdb.toString()) + "\",\n");
            w.write("  \"source_size_bytes\": " + Files.size(accdb) + ",\n");
            w.write("  \"source_sha256\": \"" + sha256(accdb) + "\",\n");
            w.write("  \"jdbc_driver\": \"io.github.spannm:ucanaccess:5.1.5\",\n");
            w.write("  \"jdbc_url_template\": \"" + escapeJson(sanitizedUrl) + "\",\n");
            w.write("  \"note\": \"Derived EDA summaries only; source Access database was not overwritten.\"\n");
            w.write("}\n");
        }
    }

    private static void writeTableInventory(Path out, List<TableInfo> tables) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            writeCsvRow(w, List.of("table_name", "table_type", "remarks", "expected_core_table"));
            for (TableInfo t : tables) {
                writeCsvRow(w, List.of(t.name, t.type, nullToEmpty(t.remarks),
                    Boolean.toString(EXPECTED_TABLES.contains(t.name))));
            }
        }
    }

    private static void writeColumnInventory(Connection conn, Path out, List<TableInfo> tables)
        throws SQLException, IOException {
        DatabaseMetaData md = conn.getMetaData();
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            writeCsvRow(w, List.of("table_name", "column_name", "type_name", "column_size",
                "nullable", "ordinal_position"));
            for (TableInfo t : tables) {
                if (!"TABLE".equalsIgnoreCase(t.type)) {
                    continue;
                }
                try (ResultSet rs = md.getColumns(null, null, t.name, "%")) {
                    while (rs.next()) {
                        writeCsvRow(w, List.of(
                            t.name,
                            rs.getString("COLUMN_NAME"),
                            rs.getString("TYPE_NAME"),
                            rs.getString("COLUMN_SIZE"),
                            rs.getString("NULLABLE"),
                            rs.getString("ORDINAL_POSITION")
                        ));
                    }
                }
            }
        }
    }

    private static void writeRowCounts(Connection conn, Path out, List<TableInfo> tables, List<String[]> errors)
        throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            writeCsvRow(w, List.of("table_name", "table_type", "row_count"));
            for (TableInfo t : tables) {
                if (!"TABLE".equalsIgnoreCase(t.type)) {
                    continue;
                }
                String sql = "SELECT COUNT(*) AS n FROM " + tableRef(t.name);
                try (Statement st = conn.createStatement();
                     ResultSet rs = st.executeQuery(sql)) {
                    rs.next();
                    writeCsvRow(w, List.of(t.name, t.type, rs.getString(1)));
                } catch (SQLException e) {
                    errors.add(new String[] {"row_count:" + t.name, sql, e.getMessage()});
                }
            }
        }
    }

    private static void writeMissingness(Connection conn, Path out, List<String[]> errors)
        throws IOException {
        List<CoreField> fields = List.of(
            new CoreField("HAUL 6-4 COMBINED", "haul_id", false),
            new CoreField("HAUL 6-4 COMBINED", "cruise_name", true),
            new CoreField("HAUL 6-4 COMBINED", "Year", false),
            new CoreField("HAUL 6-4 COMBINED", "Month", false),
            new CoreField("HAUL 6-4 COMBINED", "Day", false),
            new CoreField("HAUL 6-4 COMBINED", "date", true),
            new CoreField("HAUL 6-4 COMBINED", "sea", true),
            new CoreField("HAUL 6-4 COMBINED", "ices_rect", true),
            new CoreField("HAUL 6-4 COMBINED", "ices_division", true),
            new CoreField("HAUL 6-4 COMBINED", "shot_lat_dd", false),
            new CoreField("HAUL 6-4 COMBINED", "shot_lon_dd", false),
            new CoreField("HAUL 6-4 COMBINED", "shot_time", true),
            new CoreField("HAUL 6-4 COMBINED", "haul_time", true),
            new CoreField("HAUL 6-4 COMBINED", "shot_depth_m", true),
            new CoreField("PREDATOR 6-4 COMBINED", "pred_id", false),
            new CoreField("PREDATOR 6-4 COMBINED", "haul_id", false),
            new CoreField("PREDATOR 6-4 COMBINED", "pred", true),
            new CoreField("PREDATOR 6-4 COMBINED", "tsn", false),
            new CoreField("PREDATOR 6-4 COMBINED", "pred_length_cm", false),
            new CoreField("PREDATOR 6-4 COMBINED", "pred_wgt_g", true),
            new CoreField("PREDATOR 6-4 COMBINED", "pred_mat", true),
            new CoreField("PREDATOR 6-4 COMBINED", "pooled", true),
            new CoreField("PREDATOR 6-4 COMBINED", "num_stomachs", false),
            new CoreField("PREDATOR 6-4 COMBINED", "num_empty", false),
            new CoreField("PREDATOR 6-4 COMBINED", "mean_length_cm", false),
            new CoreField("PREDATOR 6-4 COMBINED", "tpl", true),
            new CoreField("PREY 6-4 COMBINED", "pred_id", false),
            new CoreField("PREY 6-4 COMBINED", "prey_name", true),
            new CoreField("PREY 6-4 COMBINED", "tsn", false),
            new CoreField("PREY 6-4 COMBINED", "qual_code", true),
            new CoreField("PREY 6-4 COMBINED", "prey_length", true),
            new CoreField("PREY 6-4 COMBINED", "ind_prey_wgt_g", true),
            new CoreField("PREY 6-4 COMBINED", "digestion", false),
            new CoreField("PREY 6-4 COMBINED", "min_num", false),
            new CoreField("PREY 6-4 COMBINED", "cpw", false),
            new CoreField("MURRAY_TAXONOMY", "species", true),
            new CoreField("MURRAY_TAXONOMY", "aphiaid", false),
            new CoreField("MURRAY_TAXONOMY", "adult_functional_grp", true),
            new CoreField("PROVENANCE 6-4 COMBINED", "cruise_name", true),
            new CoreField("PROVENANCE 6-4 COMBINED", "uploaded", true),
            new CoreField("PROVENANCE 6-4 COMBINED", "source_type", true),
            new CoreField("PROVENANCE 6-4 COMBINED", "data_input", true),
            new CoreField("PROVENANCE 6-4 COMBINED", "data_derived_from", true)
        );

        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            writeCsvRow(w, List.of("table_name", "field_name", "total_rows", "non_null_rows",
                "null_rows", "blank_text_rows"));
            for (CoreField f : fields) {
                String sql = "SELECT COUNT(*) AS total_rows, COUNT(" + columnRef(f.field)
                    + ") AS non_null_rows FROM " + tableRef(f.table);
                try (Statement st = conn.createStatement();
                     ResultSet rs = st.executeQuery(sql)) {
                    rs.next();
                    long total = rs.getLong("total_rows");
                    long nonNull = rs.getLong("non_null_rows");
                    long blank = f.text ? countBlankText(conn, f.table, f.field) : 0;
                    writeCsvRow(w, List.of(f.table, f.field, Long.toString(total),
                        Long.toString(nonNull), Long.toString(total - nonNull), Long.toString(blank)));
                } catch (SQLException e) {
                    errors.add(new String[] {"missingness:" + f.table + "." + f.field, sql, e.getMessage()});
                }
            }
        }
    }

    private static void writeNumericProfiles(
        Connection conn,
        Path summaryOut,
        Path histogramOut,
        List<String[]> errors
    ) throws IOException {
        List<NumericFieldSpec> specs = List.of(
            new NumericFieldSpec(
                "Predator length",
                "PREDATOR 6-4 COMBINED",
                "pred_length_cm",
                "cm",
                false
            ),
            new NumericFieldSpec(
                "Mean predator length",
                "PREDATOR 6-4 COMBINED",
                "mean_length_cm",
                "cm",
                false
            ),
            new NumericFieldSpec(
                "Stomachs represented",
                "PREDATOR 6-4 COMBINED",
                "num_stomachs",
                "count",
                true
            ),
            new NumericFieldSpec(
                "Empty stomachs",
                "PREDATOR 6-4 COMBINED",
                "num_empty",
                "count",
                true
            ),
            new NumericFieldSpec(
                "Minimum prey number",
                "PREY 6-4 COMBINED",
                "min_num",
                "count",
                true
            ),
            new NumericFieldSpec(
                "Calculated prey weight",
                "PREY 6-4 COMBINED",
                "cpw",
                "g",
                true
            )
        );

        try (
            BufferedWriter summary = Files.newBufferedWriter(summaryOut, StandardCharsets.UTF_8);
            BufferedWriter histogram = Files.newBufferedWriter(histogramOut, StandardCharsets.UTF_8)
        ) {
            writeCsvRow(summary, List.of(
                "variable", "table_name", "field_name", "unit", "non_null_rows", "null_rows",
                "zero_rows", "min", "q1", "median", "q3", "p95", "p99", "mean", "sd", "max",
                "lower_tukey_fence", "upper_tukey_fence", "below_fence_rows", "above_fence_rows"
            ));
            writeCsvRow(histogram, List.of(
                "variable", "field_name", "unit", "transform", "bin_index",
                "lower_transformed", "upper_transformed", "row_count"
            ));

            for (NumericFieldSpec spec : specs) {
                String sql = "SELECT " + columnRef(spec.field) + " AS value FROM "
                    + tableRef(spec.table) + " WHERE " + columnRef(spec.field) + " IS NOT NULL";
                try {
                    List<Double> values = new ArrayList<>();
                    try (Statement st = conn.createStatement();
                         ResultSet rs = st.executeQuery(sql)) {
                        while (rs.next()) {
                            double value = rs.getDouble("value");
                            if (!rs.wasNull() && Double.isFinite(value)) {
                                values.add(value);
                            }
                        }
                    }
                    values.sort(Double::compareTo);
                    long totalRows = countRows(conn, spec.table);
                    long nullRows = totalRows - values.size();
                    writeNumericSummary(summary, spec, values, nullRows);
                    writeNumericHistogram(histogram, spec, values, 24);
                    System.out.println("profiled " + spec.table + "." + spec.field);
                } catch (SQLException e) {
                    errors.add(new String[] {
                        "numeric_profile:" + spec.table + "." + spec.field,
                        sql,
                        e.getMessage()
                    });
                }
            }
        }
    }

    private static void writeNumericSummary(
        BufferedWriter out,
        NumericFieldSpec spec,
        List<Double> values,
        long nullRows
    ) throws IOException {
        if (values.isEmpty()) {
            writeCsvRow(out, List.of(
                spec.variable, spec.table, spec.field, spec.unit, "0", Long.toString(nullRows), "0",
                "", "", "", "", "", "", "", "", "", "", "", "0", "0"
            ));
            return;
        }

        double q1 = quantile(values, 0.25);
        double median = quantile(values, 0.50);
        double q3 = quantile(values, 0.75);
        double p95 = quantile(values, 0.95);
        double p99 = quantile(values, 0.99);
        double iqr = q3 - q1;
        double lowerFence = q1 - 1.5 * iqr;
        double upperFence = q3 + 1.5 * iqr;
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        double mean = sum / values.size();
        double squared = 0.0;
        for (double value : values) {
            double delta = value - mean;
            squared += delta * delta;
        }
        double sd = values.size() > 1 ? Math.sqrt(squared / (values.size() - 1)) : 0.0;
        long zeros = values.stream().filter(v -> v == 0.0).count();
        long below = values.stream().filter(v -> v < lowerFence).count();
        long above = values.stream().filter(v -> v > upperFence).count();

        writeCsvRow(out, List.of(
            spec.variable,
            spec.table,
            spec.field,
            spec.unit,
            Integer.toString(values.size()),
            Long.toString(nullRows),
            Long.toString(zeros),
            Double.toString(values.get(0)),
            Double.toString(q1),
            Double.toString(median),
            Double.toString(q3),
            Double.toString(p95),
            Double.toString(p99),
            Double.toString(mean),
            Double.toString(sd),
            Double.toString(values.get(values.size() - 1)),
            Double.toString(lowerFence),
            Double.toString(upperFence),
            Long.toString(below),
            Long.toString(above)
        ));
    }

    private static void writeNumericHistogram(
        BufferedWriter out,
        NumericFieldSpec spec,
        List<Double> values,
        int bins
    ) throws IOException {
        if (values.isEmpty()) {
            return;
        }
        String transform = spec.logTransform ? "signed_log10_1p" : "identity";
        double min = transformValue(values.get(0), spec.logTransform);
        double max = transformValue(values.get(values.size() - 1), spec.logTransform);
        int[] counts = new int[bins];
        if (max == min) {
            counts[0] = values.size();
        } else {
            for (double value : values) {
                double transformed = transformValue(value, spec.logTransform);
                int index = (int) Math.floor((transformed - min) / (max - min) * bins);
                counts[Math.max(0, Math.min(bins - 1, index))]++;
            }
        }
        double width = max == min ? 1.0 : (max - min) / bins;
        for (int i = 0; i < bins; i++) {
            double lower = min + i * width;
            double upper = min + (i + 1) * width;
            writeCsvRow(out, List.of(
                spec.variable,
                spec.field,
                spec.unit,
                transform,
                Integer.toString(i),
                Double.toString(lower),
                Double.toString(upper),
                Integer.toString(counts[i])
            ));
        }
    }

    private static double transformValue(double value, boolean logTransform) {
        if (!logTransform) {
            return value;
        }
        return Math.copySign(Math.log10(1.0 + Math.abs(value)), value);
    }

    private static double quantile(List<Double> sorted, double probability) {
        if (sorted.size() == 1) {
            return sorted.get(0);
        }
        double position = probability * (sorted.size() - 1);
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        if (lower == upper) {
            return sorted.get(lower);
        }
        double fraction = position - lower;
        return sorted.get(lower) + fraction * (sorted.get(upper) - sorted.get(lower));
    }

    private static long countRows(Connection conn, String table) throws SQLException {
        String sql = "SELECT COUNT(*) AS n FROM " + tableRef(table);
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getLong("n");
        }
    }

    private static String sha256(Path path) throws IOException {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 digest is unavailable", e);
        }
        try (InputStream in = Files.newInputStream(path)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) != -1) {
                digest.update(buffer, 0, read);
            }
        }
        StringBuilder hex = new StringBuilder();
        for (byte value : digest.digest()) {
            hex.append(String.format("%02x", value & 0xff));
        }
        return hex.toString();
    }

    private static long countBlankText(Connection conn, String table, String field) throws SQLException {
        String sql = "SELECT COUNT(*) AS n FROM " + tableRef(table)
            + " WHERE " + columnRef(field) + " IS NOT NULL AND " + columnRef(field) + " = ''";
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            rs.next();
            return rs.getLong("n");
        }
    }

    private static void writeQuery(Connection conn, Path out, String sql) throws SQLException, IOException {
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            writeResultSet(out, rs);
        }
    }

    private static void writeResultSet(Path out, ResultSet rs) throws SQLException, IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            ResultSetMetaData md = rs.getMetaData();
            int n = md.getColumnCount();
            List<String> header = new ArrayList<>();
            for (int i = 1; i <= n; i++) {
                header.add(md.getColumnLabel(i));
            }
            writeCsvRow(w, header);
            while (rs.next()) {
                List<String> row = new ArrayList<>();
                for (int i = 1; i <= n; i++) {
                    Object value = rs.getObject(i);
                    row.add(value == null ? "" : value.toString());
                }
                writeCsvRow(w, row);
            }
        }
    }

    private static void writeErrors(Path out, List<String[]> errors) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            writeCsvRow(w, List.of("query_name", "sql", "error"));
            for (String[] e : errors) {
                writeCsvRow(w, List.of(e[0], e[1], e[2]));
            }
        }
    }

    private static String tableRef(String name) {
        return "[" + name.replace("]", "]]") + "]";
    }

    private static String columnRef(String name) {
        return "[" + name.replace("]", "]]") + "]";
    }

    private static String compact(String sql) {
        return sql.replaceAll("\\s+", " ").trim();
    }

    private static String nullToEmpty(String value) {
        return value == null ? "" : value;
    }

    private static void writeCsvRow(BufferedWriter w, List<String> values) throws IOException {
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                w.write(",");
            }
            w.write(csvEscape(values.get(i)));
        }
        w.write("\n");
    }

    private static String csvEscape(String value) {
        if (value == null) {
            return "";
        }
        boolean quote = value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r");
        String escaped = value.replace("\"", "\"\"");
        return quote ? "\"" + escaped + "\"" : escaped;
    }

    private static String escapeJson(String value) {
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r");
    }
}
