import streamlit as st
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import PIL
import helper
import setting

def main():
    """
    Main function for the Streamlit app.
    """
    setting.configure_page()

    # Creating sidebar
    with st.sidebar:
        st.header("Image Configuration")     # Adding header to sidebar
        # Adding file uploader to sidebar for selecting images
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png"))

        # Model Options
        confidence = setting.get_model_confidence()

        # Multiselect for selecting labels
        available_labels = ['Column', 'Curtain Wall', 'Dimension', 'Door', 'Railing', 'Sliding Door', 'Stair Case', 'Wall', 'Window']
        selected_labels = setting.select_labels(available_labels)

    # Creating main page heading
    st.title("Floor Plan Object Detection using YOLOv8")

    # Creating two columns on the main page
    col1, col2 = st.columns(2)

    # Adding image to the first column if image is uploaded
    with col1:
        if source_img:
            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source_img)
            # Adding the uploaded image to the page with a caption
            st.image(source_img, caption="Uploaded Image", use_column_width=True)
        else:
            st.warning("Please upload an image.")

    model = YOLO('best.pt')

    if st.sidebar.button('Detect Objects'):
        if not source_img:
            st.warning("Please upload an image before detecting objects.")
        else:
            res = model.predict(uploaded_image, conf=confidence)
            filtered_indices = [
                i for i, box in enumerate(res[0].boxes)
                if model.names[int(box.cls)] in selected_labels
            ]
            if filtered_indices:
                filtered_data = res[0].boxes.data[filtered_indices]
                res[0].boxes = Boxes(filtered_data, res[0].orig_shape)
            else:
                res[0].boxes = Boxes(
                    torch.zeros((0, res[0].boxes.data.shape[1]), dtype=res[0].boxes.data.dtype),
                    res[0].orig_shape
                )
            filtered_boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]

            with col2:
                st.image(res_plotted, caption='Detected Image', use_column_width=True)

            # Count objects & generate CSV
            object_counts = helper.count_detected_objects(model, filtered_boxes)
            csv_file = helper.generate_csv(object_counts)

            # ── Analytics Section (منفصل تماماً تحت الصور) ────────────────
            st.divider()
            st.subheader("📊 Detection Analytics")

            if not object_counts:
                st.info("No objects detected with the selected labels.")
            else:
                total = sum(object_counts.values())

                # ── Summary metrics row ────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("🔍 Total Detections", total)
                m2.metric("🏷️ Classes Found", len(object_counts))
                avg_conf = float(filtered_boxes.conf.mean()) if len(filtered_boxes) > 0 else 0.0
                m3.metric("✅ Avg Confidence", f"{avg_conf:.1%}")

                st.write("")

                # ── Bar chart + detailed table side by side ────────────────
                chart_col, table_col = st.columns([1.2, 1])

                with chart_col:
                    st.markdown("**Object Count per Class**")
                    chart_df = pd.DataFrame({
                        "Class": list(object_counts.keys()),
                        "Count": list(object_counts.values())
                    }).sort_values("Count", ascending=False)

                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.barh(chart_df["Class"], chart_df["Count"], color="#4C9BE8")
                    ax.set_xlabel("Count")
                    ax.invert_yaxis()
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with table_col:
                    st.markdown("**Detailed Statistics**")
                    # Build per-class confidence averages
                    conf_by_class = {}
                    for box in filtered_boxes:
                        lbl = model.names[int(box.cls)]
                        conf_by_class.setdefault(lbl, []).append(float(box.conf))

                    rows = []
                    for label, count in sorted(object_counts.items(), key=lambda x: -x[1]):
                        avg = sum(conf_by_class[label]) / len(conf_by_class[label])
                        pct = count / total * 100
                        rows.append({
                            "Class": label,
                            "Count": count,
                            "Share": f"{pct:.1f}%",
                            "Avg Conf": f"{avg:.1%}"
                        })
                    st.table(pd.DataFrame(rows))

                st.write("")

                # ── Download CSV ───────────────────────────────────────────
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_file,
                    file_name='detected_objects.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()
