library(shiny)
# library(shinythemes)
library(bslib)
library(shinybusy)
library(reticulate)
library(purrr)

use_python("./env/bin/python3", required = TRUE)
source_python("globus_llama7b.py")

# Define the UI
ui <- navbarPage(
  # theme = shinytheme("cerulean"),
  theme = bs_theme(version = 4, bootswatch = "litera"),
  "SOCR GAIM",
  tabPanel(
    "Models",
    sidebarLayout(
      sidebarPanel(
        h3("Input"),
        selectInput("model", "Choose a model",
          list(
            "LLaMA 2-7B" #,
            # TODO:
            # "LLaMA 13B", "LLaMA 30B", "LLaMA 65B"
          )
        ),
        numericInput("number", label = "Number of prompts", value = 1, min = 1),
        uiOutput("entries"),
        actionButton("submit", "Submit"),
        downloadButton("download")
      ),
      mainPanel(
        h2("Output"),
        verbatimTextOutput("result")
      )
    )
  ),
  tabPanel("Features")
)

# Define the server logic
server <- function(input, output) {
  result <- "Error: No results available!"
  output$result <- renderText(
    paste0(
    "Thanks for using SOCR GAIM!\n\n",
    "Please enter your prompts and click on submit.\n\n",
    "Have fun!"
    )
  )

  # Generate text input boxes
  entry_names <- reactive(paste0("prompt", seq_len(input$number)))
  output$entries <- renderUI({
    map(entry_names(), ~textInput(.x, label = "prompt", placeholder = "Enter your prompt here"))
  })

  # Submit button
  observeEvent(input$submit, {
    # check Globus endpoint status
    connected <- endpoint_connection()
    if(connected == FALSE){
      showNotification("Error: Globus endpoint offline!", type = "error")
      return()
    }
    # retrieve prompt list
    prompts <- map(entry_names(), (function (id) input[[id]]))
    # show spinner
    show_modal_spinner(
      spin = "semipolar",
      text = paste("Running model..."),
      color = "#000000",
    )
    # run LLaMA
    result <<- run_llama7b(prompts)
    # show results
    remove_modal_spinner()
    output$result <- renderText(result)
  })

  # handle file download
  output$download <- downloadHandler(
    filename = function() {paste0("llama7b", Sys.time(), ".txt")},
    content = function(file) {writeLines(result, file)}
  )
}

# Run the Shiny app
shinyApp(ui = ui, server = server)
