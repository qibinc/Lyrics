import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { withStyles } from 'material-ui/styles';
import Typography from 'material-ui/Typography';
import TextField from 'material-ui/TextField';
import Button from 'material-ui/Button';
import Chip from 'material-ui/Chip';
import {queryWord} from "./Api";

const styles = theme => ({
    flex: {
        flex: 1,
        paddingTop: theme.spacing.unit * 2,
    },
    textField: {
        flex: 1,
        margin: theme.spacing.unit * 3,
        width: "50%",
    },
    button: {
        flex: 1,
        marginBottom: theme.spacing.unit * 3,
    },
    row: {
        display: 'flex',
        flexWrap: 'wrap',
        padding: theme.spacing.unit * 2,
    },
    chip: {
        margin: theme.spacing.unit / 2,
    }
});

class Word extends Component {

    constructor(props) {
        super(props);
        this.state = {
            query: '',
            disabled: false,
            chipData: [
            ],
        }
    }

    handleChange = name => event => {
        this.setState({
            [name]: event.target.value,
        });
    };

    handleEnterKeyPress(ev) {
        if (ev.key === 'Enter') {
            this.handleRequest();
            ev.preventDefault();
        }
    };

    handleRequest() {
        this.setState({disabled: true});
        queryWord(this.state.query)
            .then(response => {
                let prepareData = [];
                for (let idx in response) {
                    prepareData.push({key: idx, label: response[idx]});
                }
                this.setState({disabled: false, chipData: prepareData});
            });
    }

    handleChipClick(label) {
        this.setState({ query: label }, () => this.handleRequest());
    }

    render() {
        const {classes} = this.props;

        return (
            <div>
                <Typography className={classes.flex} type="title" color="inherit">
                    押韵词语
                </Typography>
                <TextField className={classes.textField}
                           id="queryWord"
                           label="请输入要查询的词"
                           margin="dense"
                           autoFocus
                           value={this.state.query}
                           onChange={this.handleChange('query')}
                           onKeyPress={(ev) => this.handleEnterKeyPress(ev)}
                           disabled={this.state.disabled}
                />
                <br/>
                <Button className={classes.button}
                        raised
                        color="primary"
                        onClick={() => this.handleRequest()}
                        disabled={this.state.disabled}
                >
                    查询
                </Button>
                <div className={classes.row}>
                    {this.state.chipData.map(data => {
                        return (
                            <Chip
                                key={data.key}
                                label={data.label}
                                className={classes.chip}
                                onClick={() => this.handleChipClick(data.label)}
                            />
                        );
                    })}
                </div>
            </div>
        )
    }
}

Word.propTypes = {
    classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(Word);
